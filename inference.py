import torch
import nibabel as nib
import numpy as np
import os
from spiking_window_swin_layer_model import SpikingSwinTransformer3D
import torch.nn.functional as F
from config import config as cfg
from spikingjelly.activation_based.encoding import PoissonEncoder
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd
    )
from monai.data import MetaTensor
from typing import List
from copy import deepcopy




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
        self.encoder = PoissonEncoder()

        # 定义 MONAI 的预处理 transform（用于每个 modality）
        self.transform = Compose([
            LoadImaged(keys=["img"], image_only=True),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=True)
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
        return self.encoder(x)



def sliding_window_inference(image, model, encoder, window_size=(128, 128, 128), stride=(64, 64, 64), T=8, num_classes=4):
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
                x_seq = encoder(patch).to(device)  # [T, C, pd, ph, pw]

                with torch.no_grad():
                    out = model(x_seq)  # [1, C, pd, ph, pw]
                    out = torch.softmax(out, dim=1)

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
    将模型输出的 TC/WT/ET 三通道 mask 转为单通道 Brats 原始标签：
    1 = ED, 2 = NET, 4 = ET
    """
    # pred: [3, D, H, W] → TC, WT, ET
    tc, wt, et = pred[0], pred[1], pred[2]  # 分别为 binary mask
    result = torch.zeros_like(tc).int()
    result[wt == 1] = 1       # Whole tumor (ED)
    result[tc == 1] = 2       # Tumor core (NET)
    result[et == 1] = 4       # Enhancing tumor (ET)
    return result

def main():
    # 路径配置
    case_dir = "./data/HGG/Brats18_2013_2_1"  # case文件夹
    case_name = os.path.basename(case_dir)

    # 模型加载
    model_ckpt = "./checkpoint/best_model_fold1.pth"  # 权重路径
    model = SpikingSwinTransformer3D(T=cfg.T).to(cfg.device)
    model.load_state_dict(torch.load(model_ckpt, map_location=cfg.device))
    model.eval()
    model.num_classes = 3 

    # 推理处理器
    preprocessor = InferencePreprocessor(modalities=cfg.modalities, T=cfg.T)

    # 图像加载与预处理
    img_tensor = preprocessor.load_case(case_dir, case_name)  # (4, D, H, W)

    # 推理
    pred_mask = sliding_window_inference(
        img_tensor, model, preprocessor.encode, window_size=cfg.patch_size, stride = (64, 64, 64)
        )

    # 恢复原始空间形状（如果pad了，裁剪回去）
    original_img_path = os.path.join(case_dir, f'{case_name}_t1.nii')
    original_img = nib.load(original_img_path)
    orig_shape = original_img.shape
    # 注意原图是 (H,W,D) 这里要对应转换
    # pred_mask 是 (D,H,W)
    d0, h0, w0 = pred_mask.shape
    # 若有pad，裁剪
    if d0 > orig_shape[2] or h0 > orig_shape[0] or w0 > orig_shape[1]:
        pred_mask = pred_mask[:orig_shape[2], :orig_shape[0], :orig_shape[1]]

    # 转置到 (H,W,D)
    pred_mask = np.transpose(pred_mask, (1, 2, 0))

    # 保存预测结果
    pred_nii = nib.Nifti1Image(pred_mask.astype(np.uint8), original_img.affine, original_img.header)
    save_path = os.path.join(case_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, save_path)
    print(f"预测mask保存至: {save_path}")

if __name__ == "__main__":
    main()
