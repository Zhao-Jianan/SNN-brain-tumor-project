import os, traceback
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConvertToMultiChannelBasedOnBratsClassesd,
    RandSpatialCropd, RandFlipd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd,
    ToTensord, Orientationd, Spacingd
)
from monai.data import Dataset as MonaiDataset
from monai.transforms.utils import allow_missing_keys_mode
from spikingjelly.clock_driven.encoding import PoissonEncoder, LatencyEncoder


class BraTSDataset(MonaiDataset):
    def __init__(self, data_dicts, T=8, patch_size=(128,128,128), num_classes=4, mode="train", encode_method='poisson', debug=False):
        """
        data_dicts: list of dict, 每个 dict 形如：
          {
            "image": [t1_path, t1ce_path, t2_path, flair_path],  # 四模态路径列表
            "label": label_path
          }
        """
        self.data_dicts = data_dicts
        self.T = T
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.mode = mode
        self.debug = debug
        self.encode_method = encode_method
        self.poisson_encoder = PoissonEncoder()
        self.latency_encoder = LatencyEncoder(self.T)

        # 读取数据，自动封装成 MetaTensor (带affine)
        self.load_transform = Compose([
            LoadImaged(keys=["image", "label"]),  # 加载 nii，自动带 affine
            EnsureChannelFirstd(keys=["image", "label"]),  # 保证通道维度在前 (C, D, H, W)
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),  # [0: TC, 1: WT, 2: ET]
        ])

        # 统一空间预处理
        self.preprocess = Compose([
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ])

        # 数据增强 pipeline
        self.train_transform = Compose([
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"])
        ])

        self.val_transform = Compose([
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"])
        ])

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data = self.data_dicts[idx]
        case_name = os.path.basename(data["label"]).replace("_seg.nii", "")

        data = self.load_transform(data)  # load nii -> MetaTensor (C, D, H, W) + affine

        img_meta = data["image"].meta
        label_meta = data["label"].meta
        
        # spacing
        img_spacing = img_meta.get("pixdim", None)
        label_spacing = label_meta.get("pixdim", None)
        
        need_orientation_or_spacing = False

        if img_meta.get("spatial_shape") is None:  # 安全性检查
            need_orientation_or_spacing = True
        else:
            # 检查 spacing 是否不是 (1.0, 1.0, 1.0)
            if not (torch.allclose(torch.tensor(img_spacing[:3]), torch.tensor([1.0, 1.0, 1.0])) and
                    torch.allclose(torch.tensor(label_spacing[:3]), torch.tensor([1.0, 1.0, 1.0]))):
                need_orientation_or_spacing = True
            # 检查 orientation 是否不是 RAS
            if not (img_meta.get("original_channel_dim", None) == 0 and
                    img_meta.get("original_affine", None) is not None):
                need_orientation_or_spacing = True

        # 只有在必要时运行 preprocess
        if need_orientation_or_spacing:
            print(f'DO PREPROPROCESS!!!')
            data = self.preprocess(data)


        data = self.patch_crop(data)

        if self.mode == "train":
            data = self.train_transform(data)
        else:
            data = self.val_transform(data)

        img = data["image"]  # Tensor (C, D, H, W)
        label = data["label"]  # Tensor (C_label, D, H, W) 
        
        if self.debug:
            unique_vals = torch.unique(label)
            if label.min() < 0 or label.max() >= self.num_classes:
                print(f"[ERROR] Label out of range in sample {case_name}")
                print(f"Label unique values: {unique_vals}")
                raise ValueError(f"Label contains invalid class ID(s): {unique_vals.tolist()}")
            
            if img.dim() == 4:
                C = img.shape[0]
                for c in range(C):
                    min_val = img[c].min().item()
                    max_val = img[c].max().item()
                    print(f"Channel {c}: min={min_val:.4f}, max={max_val:.4f}")
            else:
                print("Not a 4D tensor; skipping per-channel stats.")

        # 生成 T 个时间步的脉冲输入，重复编码
        img_rescale = self.rescale_to_unit_range(img)
        
        if self.encode_method == 'poisson':
            x_seq = torch.stack([self.poisson_encoder(img_rescale) for _ in range(self.T)], dim=0)
        elif self.encode_method == 'latency':
            img_rescale = img_rescale.unsqueeze(0)  # (1,C,D,H,W)
            self.latency_encoder.encode(img_rescale)  # (T,1,C,D,H,W)
            spike = self.latency_encoder.spike
            x_seq = spike.squeeze(1)  # (T,C,D,H,W)
            
        else:
            raise NotImplementedError(f"Encoding method '{self.encode_method}' is not implemented.")
        # x_seq: (T, C, D, H, W), label: (C_label, D, H, W)
        return x_seq, label

    
    def patch_crop(self, data):
        img = data["image"]        # (C, D, H, W)
        label = data["label"]      # (C, D, H, W), one-hot

        _, D, H, W = img.shape
        pd, ph, pw = self.patch_size
        assert D >= pd and H >= ph and W >= pw, f"Patch size {self.patch_size} too big for image {img.shape}"

        # 合并 one-hot 通道，得到 (D, H, W) 的前景掩码
        foreground_mask = label.sum(axis=0) > 0     # tensor 仍然，torch.bool

        # 转 numpy 并找到非零索引坐标，结果形状是 (N, 3)
        foreground = np.argwhere(foreground_mask.cpu().numpy())

        if len(foreground) > 0:
            center = foreground[np.random.choice(len(foreground))]
            cd, ch, cw = center
        else:
            d_start = np.random.randint(0, D - pd + 1)
            h_start = np.random.randint(0, H - ph + 1)
            w_start = np.random.randint(0, W - pw + 1)

            data["image"] = img[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
            data["label"] = label[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
            return data

        # 计算 patch 起始位置（防越界）
        d_start = np.clip(cd - pd // 2, 0, D - pd)
        h_start = np.clip(ch - ph // 2, 0, H - ph)
        w_start = np.clip(cw - pw // 2, 0, W - pw)

        data["image"] = img[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        data["label"] = label[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]

        return data
    
    def rescale_to_unit_range(self, x: torch.Tensor) -> torch.Tensor:
        # 逐个样本 min-max 归一化，不改变整体分布，只用于编码器
        x_min = x.amin(dim=[1, 2, 3], keepdim=True)
        x_max = x.amax(dim=[1, 2, 3], keepdim=True)
        x_rescaled = (x - x_min) / (x_max - x_min + 1e-8)
        return x_rescaled.clamp(0., 1.)
