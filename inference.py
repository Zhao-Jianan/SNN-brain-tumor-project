import torch
import nibabel as nib
import numpy as np
import os
from spiking_swin_unet_model import SpikingSwinUNet3D
import torch.nn.functional as F
from config import config as cfg
from spikingjelly.activation_based.encoding import PoissonEncoder
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord
    )
from copy import deepcopy
import time
from scipy.ndimage import binary_dilation, binary_opening, generate_binary_structure
from inference_helper import TemporalSlidingWindowInference


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
    data["image"] = data["image"].permute(0, 3, 1, 2)
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
    对BraTS预测标签做形态学后处理:
    - ET (4) 膨胀1个像素
    - NCR/NET (1) 保持原状或开运算平滑
    - ED (2) 保持原状
    输入：
        pred_mask: (H, W, D) ndarray, uint8，标签值为0,1,2,4
    返回：
        后处理后的标签mask，shape相同
    """
    structure = generate_binary_structure(rank=3, connectivity=1)  # 3D结构元素，邻接6个方向

    # 分离各标签mask
    et_mask = (pred_mask == 4)
    ncr_mask = (pred_mask == 1)
    edema_mask = (pred_mask == 2)
    print("Before Postprocessing:")
    print("Sum ET:", np.sum(et_mask))
    print("Sum ED:", np.sum(edema_mask))
    print("Sum NCR:", np.sum(ncr_mask))

    # ET膨胀1个像素
    et_dilated = binary_dilation(et_mask, structure=structure, iterations=1)
    et_dilated &= (edema_mask | et_mask) 

    # NCR保持原状，或者用开运算平滑（可选）
    # ncr_processed = binary_opening(ncr_mask, structure=structure, iterations=1)
    ncr_processed = ncr_mask  # 保持原状

    # 初始化新的mask
    new_mask = np.zeros_like(pred_mask, dtype=np.uint8)

    # 按优先级分配：ET > ED > NCR
    new_mask[et_dilated] = 4
    new_mask[(edema_mask) & (new_mask == 0)] = 2
    new_mask[(ncr_processed) & (new_mask == 0)] = 1

    # 背景默认是0，不用赋值
    print("Postprocessing results:")
    print("Sum ET:", np.sum(new_mask == 4))
    print("Sum ED:", np.sum(new_mask == 2))
    print("Sum NCR:", np.sum(new_mask == 1))
    return new_mask


def inference_single_case(case_dir, model, inference_engine, device, T=8):
    # 用cfg.modalities拼4模态路径
    case_name = os.path.basename(case_dir)
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]

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

    out_path = os.path.join(case_dir, f"{case_name}_pred_mask.nii")
    nib.save(pred_nii, out_path)
    print(f"Saved prediction: {out_path}")

    

def main():
    case_dir = "./data/HGG/Brats18_2013_27_1"
    model_ckpt = "./checkpoint/tumor_center_crop_best_model_fold1.pth"

    model = SpikingSwinUNet3D(window_size=cfg.window_size, T=cfg.T, step_mode=cfg.step_mode).to(cfg.device)  # 模型.to(cfg.device)
    model.load_state_dict(torch.load(model_ckpt, map_location=cfg.device))
    model.eval()

    inference_engine = TemporalSlidingWindowInference(
        patch_size=cfg.patch_size,
        overlap=0.5, # cfg.overlap
        sw_batch_size=1,
        encode_method=cfg.encode_method,
        T= 10, # cfg.T
        num_classes=cfg.num_classes
    )
    
    inference_single_case(case_dir, model, inference_engine, cfg.device, T=cfg.T)

    

if __name__ == "__main__":
    main()