import torch
import nibabel as nib
import numpy as np
import os
from spiking_swin_unet_model_4layer_no_dropout import SpikingSwinUNet3D
import torch.nn.functional as F
from config import config as cfg
from spikingjelly.activation_based.encoding import PoissonEncoder
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd
    )
from monai.data import MetaTensor
from typing import List
from copy import deepcopy
import time
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

class PoissonEncoderWrapper:
    def __init__(self, T):
        self.T = T
        self.encoder = PoissonEncoder()

    def __call__(self, img):
        # img: Tensor [C, D, H, W]
        # 返回Tensor [T, C, D, H, W]
        return torch.stack([self.encoder(img) for _ in range(self.T)], dim=0)



class InferencePreprocessor:
    def __init__(self, modalities: List[str], T: int):
        """
        :param modalities: 模态名称列表，如 ['t1', 't1ce', 't2', 'flair']
        :param T: 时间步数，用于泊松编码
        """
        self.modalities = modalities
        self.T = T
        self.encoder = PoissonEncoderWrapper(T)

        # 定义 MONAI 的预处理 transform（用于每个 modality）
        self.transform = Compose([
            LoadImaged(keys=["img"], image_only=True),
            EnsureChannelFirstd(keys=["img"]),
            # NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=True)
        ])

    def load_case(self, case_dir: str, case_name: str) -> torch.Tensor:
        """
        加载并归一化多模态 MRI 图像，返回 tensor (C, D, H, W)，并保留 affine
        """
        imgs = []
        for mod in self.modalities:
            path = os.path.join(case_dir, f'{case_name}_{mod}.nii')
            data_dict = {"img": path}
            normalized = self.transform(deepcopy(data_dict))["img"]  # (1, D, H, W)
            img = normalized.squeeze(0).numpy()  # 去掉 channel 维度 (D, H, W)
            img = np.transpose(img, (0, 1, 2))  # 保持 D, H, W
            imgs.append(img)
        
        imgs = np.stack(imgs, axis=0)  # (4, D, H, W)
        return torch.tensor(imgs).float()

    def rescale_to_unit_range(self, x: torch.Tensor) -> torch.Tensor:
        """
        将每个样本缩放到 0–1，按通道分别处理。用于编码器输入。
        输入 x: Tensor [C, D, H, W]
        """
        x_min = x.amin(dim=[1, 2, 3], keepdim=True)
        x_max = x.amax(dim=[1, 2, 3], keepdim=True)
        x_rescaled = (x - x_min) / (x_max - x_min + 1e-8)
        return x_rescaled.clamp(0., 1.)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rescale_to_unit_range(x)
        return self.encoder(x)  # 返回的是 [T, C, D, H, W]


def zscore_patch(patch: torch.Tensor) -> torch.Tensor:
    """
    对每个 patch 按通道做 z-score 标准化。
    输入 patch: Tensor [C, D, H, W]
    """
    patch = patch.clone()
    for c in range(patch.shape[0]):
        nonzero = patch[c][patch[c] != 0]
        if nonzero.numel() > 0:
            mean = nonzero.mean()
            std = nonzero.std()
            patch[c] = (patch[c] - mean) / (std + 1e-8)
        else:
            patch[c] = 0  # 全零通道则不处理
    return patch

def sliding_window_inference(image, model, encoder, window_size=(128, 128, 128), stride=(64, 64, 64), num_classes=3):
    """
    Sliding window inference for SNN segmentation
    """
    device = next(model.parameters()).device
    model.eval()

    C, D, H, W = image.shape
    pd, ph, pw = window_size
    sd, sh, sw = stride

    output_probs = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((1, D, H, W), dtype=torch.float32, device=device)

    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                patch = image[:, d:d+pd, h:h+ph, w:w+pw]
                # 对 patch 做 z-score normalization（按通道）
                patch = zscore_patch(patch)
                x_seq = encoder(patch).to(device)  # [T, C, pd, ph, pw]

                with torch.no_grad():
                    x_seq = x_seq.unsqueeze(1)
                    out = model(x_seq)  # [1, C, pd, ph, pw]
                    out = torch.sigmoid(out)

                output_probs[:, d:d+pd, h:h+ph, w:w+pw] += out.squeeze(0)
                count_map[:, d:d+pd, h:h+ph, w:w+pw] += 1

    count_map[count_map == 0] = 1
    output_probs /= count_map
    # 二值化每个通道
    binary_pred = (output_probs > 0.5).int().cpu()  # [3, D, H, W]

    # 转换为单通道 BraTS 标签格式
    pred = convert_prediction_to_label(binary_pred)  # [D, H, W]

    return pred


def convert_prediction_to_label(pred: torch.Tensor) -> torch.Tensor:
    """
    BraTS标签转换，输入 pred顺序：TC, WT, ET
    """
    tc, wt, et = pred[0], pred[1], pred[2]
    print("Sum TC:", pred[0].sum().item())
    print("Sum WT:", pred[1].sum().item())
    print("Sum ET:", pred[2].sum().item())

    result = torch.zeros_like(tc, dtype=torch.int32)

    # ET赋值4
    result[et == 1] = 4

    # TC去除ET赋1
    tc_core = (tc == 1) & (et == 0)
    result[tc_core] = 1

    # WT去除TC和ET赋2
    edema = (wt == 1) & (tc == 0) & (et == 0)
    result[edema] = 2

    return result




def postprocess_brats_label(pred_mask: np.ndarray) -> np.ndarray:
    """
    对BraTS预测标签做形态学后处理:
    - ET (4) 膨胀
    - NCR/NET (1) 腐蚀
    - 其他保持不变
    输入：
        pred_mask: (H, W, D) ndarray, uint8，标签值为0,1,2,4
    返回：
        后处理后的标签mask，shape相同
    """
    structure = generate_binary_structure(rank=3, connectivity=1)  # 3D 结构元素，邻接6个方向
    
    # 分离各标签
    et_mask = (pred_mask == 4)
    ncr_mask = (pred_mask == 1)
    edema_mask = (pred_mask == 2)

    # ET膨胀，膨胀1个像素
    et_dilated = binary_dilation(et_mask, structure=structure, iterations=1)

    # NCR/NET腐蚀，腐蚀1个像素
    ncr_eroded = binary_erosion(ncr_mask, structure=structure, iterations=1)

    # 合成新的mask，优先级 ET > NCR > ED
    new_mask = np.zeros_like(pred_mask)
    new_mask[et_dilated] = 4
    # 只在非ET区域赋NCR，避免腐蚀后越界覆盖ET
    new_mask[(ncr_eroded) & (~et_dilated)] = 1
    # ED只赋非ET非NCR区域
    new_mask[(edema_mask) & (~et_dilated) & (~ncr_eroded)] = 2

    # 其余部分为0（背景）
    return new_mask




def main():
    # 路径配置
    case_dir = "./data/HGG/Brats18_CBICA_AAB_1"  # case文件夹
    case_name = os.path.basename(case_dir)

    # 模型加载
    model_ckpt = "./checkpoint/brats-18-possion-nodropout-T10/best_model_fold4.pth"  # 权重路径
    model = model = SpikingSwinUNet3D(window_size=cfg.window_size, T=cfg.T, step_mode=cfg.step_mode).to(cfg.device)  # 模型.to(cfg.device)
    model.load_state_dict(torch.load(model_ckpt, map_location=cfg.device))
    model.eval()
    model.num_classes = 3 

    # 推理处理器
    preprocessor = InferencePreprocessor(modalities=cfg.modalities, T=cfg.T)

    # 图像加载与预处理
    img_tensor = preprocessor.load_case(case_dir, case_name)  # (4, D, H, W)

    # 推理
    start_time = time.time()
    pred_mask = sliding_window_inference(
        img_tensor, model, preprocessor.encode, window_size=cfg.patch_size, stride = (64, 64, 64)
        )
    
    # 计时结束
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.2f} seconds")

    # 恢复原始空间形状（如果pad了，裁剪回去）
    original_img_path = os.path.join(case_dir, f'{case_name}_t1.nii')
    original_img = nib.load(original_img_path)
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    # pred_mask 是 (D, H, W)
    h0, w0, d0 = pred_mask.shape
    h_orig, w_orig, d_orig = original_img.shape
    
    # 如果有pad，裁剪回原图大小
    if h0 > h_orig or w0 > w_orig or d0 > d_orig:
        pred_mask = pred_mask[:h_orig, :w_orig, :d_orig]
        
        
    # 后处理
    pred_mask = postprocess_brats_label(pred_mask)

    # 保存预测结果
    pred_mask = pred_mask.astype(np.uint8)
    pred_nii = nib.Nifti1Image(pred_mask, original_img.affine, original_img.header)
    save_path = os.path.join(case_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, save_path)
    print(f"预测mask保存至: {save_path}")


if __name__ == "__main__":
    main()
