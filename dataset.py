import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from spikingjelly.activation_based.encoding import PoissonEncoder



class BraTSDataset(Dataset):
    def __init__(self, case_dirs, T=8, transform=None, patch_size=(128, 128, 64), num_classes=4, debug=False):
        self.case_dirs = case_dirs
        self.T = T
        self.transform = transform
        self.patch_size = patch_size
        self.encoder = PoissonEncoder()
        self.num_classes = num_classes
        self.debug = debug

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_name = os.path.basename(case_dir)

        # 4个模态路径
        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = []
        for mod in modalities:
            path = os.path.join(case_dir, f'{case_name}_{mod}.nii')
            img = nib.load(path).get_fdata()
            
            # BraTS per-image Z-score normalization on non-zero voxels only
            nonzero = img > 0
            if np.any(nonzero):
                mean = img[nonzero].mean()
                std = img[nonzero].std()
                img[nonzero] = (img[nonzero] - mean) / (std + 1e-8)
            else:
                img[:] = 0  # Handle rare all-zero image case


            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)

        # 拼接4个模态，形状变为 (4, D, H, W)
        img = np.stack(imgs, axis=0)
        img = torch.tensor(img).float()

        # 加载标签
        label_path = os.path.join(case_dir, f'{case_name}_seg.nii')
        label = nib.load(label_path).get_fdata()
        label = np.transpose(label, (2, 0, 1))  # (D, H, W)
        label = torch.tensor(label).long()

        label[label == 4] = 3

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        # 裁剪patch
        img, label = self.patch_crop(img, label)

        if self.debug:
            unique_vals = torch.unique(label)
            if label.min() < 0 or label.max() >= self.num_classes:
                print(f"[ERROR] Label out of range in sample {case_name}")
                print(f"Label unique values: {unique_vals}")
                raise ValueError(f"Label contains invalid class ID(s): {unique_vals.tolist()}")

        # Poisson编码，x_seq形状 [T, 4, D, H, W]
        x_seq = torch.stack([self.encoder(img) for _ in range(self.T)], dim=0)

        return x_seq, label

    def patch_crop(self, img, label):
        _, D, H, W = img.shape
        pd, ph, pw = self.patch_size
        assert D >= pd and H >= ph and W >= pw, \
            f"Patch size {self.patch_size} too big for image {img.shape}"

        # 获取前景索引（label > 0 的体素坐标）
        foreground = np.argwhere(label.numpy() > 0)

        if len(foreground) > 0:
            # 从前景中随机选择一个中心点
            center = foreground[np.random.choice(len(foreground))]
            cd, ch, cw = center

            # 计算 patch 起始位置（注意防止越界）
            d_start = np.clip(cd - pd // 2, 0, D - pd)
            h_start = np.clip(ch - ph // 2, 0, H - ph)
            w_start = np.clip(cw - pw // 2, 0, W - pw)
        else:
            # 若无前景，退回到随机裁剪（纯背景 patch）
            d_start = np.random.randint(0, D - pd + 1)
            h_start = np.random.randint(0, H - ph + 1)
            w_start = np.random.randint(0, W - pw + 1)

        img_patch = img[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        label_patch = label[d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]

        return img_patch, label_patch