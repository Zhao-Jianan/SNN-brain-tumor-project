import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from spikingjelly.activation_based.encoding import PoissonEncoder



class BraTSDataset(Dataset):
    def __init__(self, case_dirs, T=8, transform=None, patch_size=(128, 128, 128), num_classes=4, debug=False):
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
        """
        根据索引获取病例数据，包含图像读取、处理和编码
        """
        case_dir = self.case_dirs[idx]
        case_name = os.path.basename(case_dir)

        img = self.load_modalities(case_dir, case_name)   # 加载图像
        label = self.load_label(case_dir, case_name)      # 加载标签

        if self.transform:
            img = self.transform(img)                      # 图像变换
            label = self.transform(label)                  # 标签变换

        img, label = self.patch_crop(img, label)          # 随机裁剪patch

        if self.debug:
            unique_vals = torch.unique(label)
            # 检查标签值是否合法
            if label.min() < 0 or label.max() >= self.num_classes:
                print(f"[ERROR] Label out of range in sample {case_name}")
                print(f"Label unique values: {unique_vals}")
                raise ValueError(f"Label contains invalid class ID(s): {unique_vals.tolist()}")

        # Poisson编码生成T个时间步的输入序列
        x_seq = torch.stack([self.encoder(img) for _ in range(self.T)], dim=0)

        return x_seq, label


    def normalize_nonzero(self, img):
        """对非零体素进行Z-score标准化"""
        nonzero = img > 0
        if np.any(nonzero):
            mean = img[nonzero].mean()  # 计算非零像素均值
            std = img[nonzero].std()  # 计算非零像素标准差
            img[nonzero] = (img[nonzero] - mean) / (std + 1e-8)  # 标准化
        else:
            img[:] = 0  # 若全零则置零处理
        return img


    def load_modalities(self, case_dir, case_name):
        """
        读取并标准化4个模态图像，返回Tensor格式数据
        """
        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = []
        for mod in modalities:
            path = os.path.join(case_dir, f'{case_name}_{mod}.nii')
            img = nib.load(path).get_fdata()    # 读取nii数据
            img = np.transpose(img, (2, 0, 1))   # 转换维度到(D, H, W)
            img = self.normalize_nonzero(img)   # 标准化非零区域
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)       # 合并为(4, D, H, W)
        imgs = torch.tensor(imgs).float()  # 转换为Tensor
        return imgs


    def load_label(self, case_dir, case_name):
        """
        读取标签数据并进行处理，返回LongTensor格式
        """
        label_path = os.path.join(case_dir, f'{case_name}_seg.nii')
        label = nib.load(label_path).get_fdata()   # 读取标签
        label = np.transpose(label, (2, 0, 1))    # 转换维度(D, H, W)
        label = torch.tensor(label).long()     # 转为LongTensor
        label[label == 4] = 3                  # 标签4映射到3
        return label
    

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