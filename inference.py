import torch
import nibabel as nib
import numpy as np
import os
from spiking_swin_unet_model import SpikingSwinUNet3D
import torch.nn.functional as F
from config import config as cfg
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord
    )
from copy import deepcopy
import time
from scipy.ndimage import binary_dilation, binary_opening, label, generate_binary_structure
from inference_helper import TemporalSlidingWindowInference
from tqdm import tqdm


def preprocess_for_inference(image_paths, T=8):
    """
    image_paths: list of 4 modality paths [t1, t1ce, t2, flair]
    
    Returns:
        x_seq: torch.Tensor, shape (T, C, D, H, W)
    """
    data_dict = {"image": image_paths}
    
    # Step 1: Load + Channel First
    load_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ])
    data = load_transform(data_dict)
    data["image"] = data["image"].permute(0, 3, 1, 2).contiguous()
    print("Loaded image shape:", data["image"].shape)  # (C, D, H, W)
    
    img_meta = data["image"].meta
    img_spacing = img_meta.get("pixdim", None)

    # Step 2: Spatial Normalization (Orientation + Spacing)
    need_orientation_or_spacing = False
    if img_meta.get("spatial_shape") is None:
        need_orientation_or_spacing = True
    else:
        if not torch.allclose(torch.tensor(img_spacing[:3]), torch.tensor([1.0, 1.0, 1.0])):
            need_orientation_or_spacing = True
        if not (img_meta.get("original_channel_dim", None) == 0 and img_meta.get("original_affine", None) is not None):
            need_orientation_or_spacing = True
    
    if need_orientation_or_spacing:
        print("DO PREPROCESS!!!")
        preprocess = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ])
        data = preprocess(data)
    
    # Step 3: Intensity Normalization
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    data = normalize(data)
    
    # Step 4: ToTensor
    to_tensor = ToTensord(keys=["image"])
    data = to_tensor(data)
    
    # Step 5: Repeat T times to add temporal dimension
    img = data["image"]  # shape: (C, D, H, W)
    img_seq = img.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1, 1, 1)  # (T, B=1, C, D, H, W)
    
    return img_seq



def convert_prediction_to_label(pred: torch.Tensor) -> torch.Tensor:
    """
    BraTS标签转换，输入 pred顺序：TC, WT, ET
    """
    tc, wt, et = pred[0], pred[1], pred[2]

    result = torch.zeros_like(tc, dtype=torch.int32)

    # ET赋4
    result[et == 1] = 4

    # TC赋1，排除ET
    tc_only = (tc == 1) & (et == 0)
    result[tc_only] = 1

    # ED = WT - (TC + ET)
    edema_only = (wt == 1) & (tc == 0) & (et == 0)
    result[edema_only] = 2
    
    print("Sum TC:", tc.sum().item())
    print("Sum WT:", wt.sum().item())
    print("Sum ET:", et.sum().item())
    print("Result summary:")
    print("Sum NCR:", (result == 1).sum().item())
    print("Sum ED:", (result == 2).sum().item())
    print("Sum ET:", (result == 4).sum().item())

    return result


def postprocess_brats_label(pred_mask: np.ndarray) -> np.ndarray:
    """
    BraTS预测标签后处理：
    - ET (4): 向外扩张一圈，只吸收外部的NCR和ED，不吞噬ET内部的NCR
    - NCR/NET (1): 外部NCR做开运算，ET内部NCR保持原样
    - ED (2): 保持原状
    """

    structure = generate_binary_structure(3, 1)

    # 原始标签
    et_mask = (pred_mask == 4)
    ncr_mask = (pred_mask == 1)
    edema_mask = (pred_mask == 2)

    print("Before Postprocessing:")
    print("Sum ET:", np.sum(et_mask))
    print("Sum ED:", np.sum(edema_mask))
    print("Sum NCR:", np.sum(ncr_mask))

    # Step 1: 分离ET内部的NCR（要保护的）与ET外部的NCR（可处理的）
    ncr_inside_et = ncr_mask & et_mask
    ncr_outside_et = ncr_mask & (~et_mask)

    # Step 2: 对外部NCR做开运算
    ncr_outside_processed = binary_opening(ncr_outside_et, structure=structure, iterations=1)
    ncr_processed = ncr_outside_processed | ncr_inside_et

    # 被剥掉的外部NCR边缘
    ncr_removed = ncr_outside_et & (~ncr_outside_processed)

    # Step 3: 构造ET的“外壳”：从ET外面包一圈，不含ET原始区域
    et_outer_shell = binary_dilation(et_mask, structure=structure, iterations=1) & (~et_mask)

    # Step 4: 只允许ET扩张到其“外壳”中满足条件的区域（NCR外部边缘 or ED）
    et_expand_target = et_outer_shell & (ncr_removed | edema_mask)

    # Step 5: 最终ET = 原始ET + 允许扩张区域（外壳目标）
    et_final = et_mask | et_expand_target

    # Step 6: 构建最终mask
    new_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    new_mask[et_final] = 4
    new_mask[(edema_mask) & (new_mask == 0)] = 2
    new_mask[(ncr_processed) & (new_mask == 0)] = 1

    # Step 7: 剩余被删掉的NCR边缘如果未被覆盖，强制转ET（避免留背景）
    ncr_remaining = ncr_removed & (new_mask == 0)
    new_mask[ncr_remaining] = 4

    print("Postprocessing results:")
    print("Sum ET:", np.sum(new_mask == 4))
    print("Sum ED:", np.sum(new_mask == 2))
    print("Sum NCR:", np.sum(new_mask == 1))

    return new_mask





def pred_single_case(case_dir, inference_dir, model, inference_engine, device, T=8):
    # 用cfg.modalities拼4模态路径
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]
    print("Image paths:", image_paths)
    # image_paths = [os.path.join(case_dir, f"{mod}.nii.gz") for mod in cfg.modalities]

    # 预处理，返回 (T, C, D, H, W) Tensor
    x_seq = preprocess_for_inference(image_paths, T=T)
    x_seq = x_seq.to(device)

    
    start_time = time.time()
    # sliding window推理
    with torch.no_grad():
        output = inference_engine(x_seq, model)
        
    # 计时结束
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.2f} seconds")

    # 阈值二值化，模型输出已mean
    output_prob = torch.sigmoid(output)
    output_bin = (output_prob > 0.5).int().squeeze(0)

    # 转换标签格式，后处理
    label_raw = convert_prediction_to_label(output_bin)
    label_np = label_raw.cpu().numpy().astype(np.uint8)
    label_np = np.transpose(label_np, (1, 2, 0))
    print("Label shape before postprocessing:", label_np.shape)  # (H, W, D)
    # label_np = postprocess_brats_label(label_np)
    print("Saved label shape:", label_np.shape)  # (H, W, D)

    # 以t1ce为参考保存nii
    ref_nii = nib.load(image_paths[cfg.modalities.index('t1ce')])
    pred_nii = nib.Nifti1Image(label_np, affine=ref_nii.affine, header=ref_nii.header)

    out_path = os.path.join(inference_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, out_path)
    print(f"Saved prediction: {out_path}")



def run_inference_on_case(case_dir: str, save_dir: str, model, inference_engine, device, T: int):
    os.makedirs(save_dir, exist_ok=True)
    pred_single_case(case_dir, save_dir, model, inference_engine, device, T=T)


def run_inference_on_folder(case_root: str, save_dir: str, model, inference_engine, device, T: int):
    """
    推理多个 case 文件夹
    """
    os.makedirs(save_dir, exist_ok=True)
    case_dirs = sorted([
        os.path.join(case_root, name) for name in os.listdir(case_root)
        if os.path.isdir(os.path.join(case_root, name))
    ])
    print(f"Found {len(case_dirs)} cases to infer.")
    
    for case_dir in tqdm(case_dirs, desc="Batch Inference"):
        run_inference_on_case(case_dir, save_dir, model, inference_engine, device, T)


    

def main():
    case_dir = "./data/HGG/Brats18_2013_27_1"
    model_ckpt = "./checkpoint/best_model_fold_inference.pth"
    inference_dir = "./pred"
    
    model = SpikingSwinUNet3D(window_size=cfg.window_size, T=cfg.T, step_mode=cfg.step_mode).to(cfg.device)  # 模型.to(cfg.device)
    model.load_state_dict(torch.load(model_ckpt, map_location=cfg.device))
    model.eval()

    inference_engine = TemporalSlidingWindowInference(
        patch_size=cfg.patch_size,
        overlap=cfg.overlap, # cfg.overlap
        sw_batch_size=1,
        mode="constant", # "gaussian", "constant"
        encode_method=cfg.encode_method,
        T= cfg.T, # cfg.T
        num_classes=cfg.num_classes
    )
    
    if os.path.isdir(case_dir) and any(os.path.isdir(os.path.join(case_dir, f)) for f in os.listdir(case_dir)):
        print(f"Running batch inference")
        run_inference_on_folder(case_dir, inference_dir, model, inference_engine, cfg.device, cfg.T)
    else:
        print(f"Running single inference")
        run_inference_on_case(case_dir, inference_dir, model, inference_engine, cfg.device, cfg.T)

    

if __name__ == "__main__":
    main()