import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from spikingjelly.activation_based.encoding import PoissonEncoder
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import KFold
from medpy.metric import binary


class BraTSDataset(Dataset):
    def __init__(self, case_dirs, T=2, transform=None, patch_size=(128, 128, 64), num_classes=4, debug=False):
        self.case_dirs = case_dirs
        self.T = T
        self.transform = transform
        self.patch_size = patch_size
        self.encoder = PoissonEncoder()
        self.num_classes = num_classes
        self.debug = debug  # 是否启用调试模式

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_name = os.path.basename(case_dir)

        t1ce_path = os.path.join(case_dir, f'{case_name}_t1ce.nii')
        label_path = os.path.join(case_dir, f'{case_name}_seg.nii')

        # 加载 MRI 图像和标签
        img = nib.load(t1ce_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # 归一化图像
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 转换为 tensor
        img = torch.tensor(img).unsqueeze(0).float()  # [1, D, H, W]
        label = torch.tensor(label).long()            # [D, H, W]

        uniques = np.unique(label)
        for u in uniques:
            if u not in [0, 1, 2, 4]:
                raise RuntimeError('unexpected label')
        
        # 标签映射：将 4 -> 3
        label[label == 4] = 3

        # 可选 transform（注意 label 不应该做旋转、缩放等不一致变换）
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        # 随机裁剪 Patch
        img, label = self.patch_crop(img, label)

        # Debug: 检查标签是否合法
        if self.debug:
            unique_vals = torch.unique(label)
            if label.min() < 0 or label.max() >= self.num_classes:
                print(f"[ERROR] Label out of range in sample {case_name}")
                print(f"Label unique values: {unique_vals}")
                raise ValueError(f"Label contains invalid class ID(s): {unique_vals.tolist()}")

        # 使用 Poisson 编码：[T, 1, D, H, W]
        x_seq = torch.stack([self.encoder(img) for _ in range(self.T)], dim=0)
        print('Done')
        return x_seq, label

    def patch_crop(self, img, label):
        _, D, H, W = img.shape
        pd, ph, pw = self.patch_size

        assert D >= pd and H >= ph and W >= pw, \
            f"Patch size {self.patch_size} too big for image {img.shape}"

        d_start = np.random.randint(0, D - pd + 1)
        h_start = np.random.randint(0, H - ph + 1)
        w_start = np.random.randint(0, W - pw + 1)

        img_patch = img[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label_patch = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        return img_patch, label_patch
    

root_dir = './data/HGG'
case_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]


train_dataset = BraTSDataset(
    case_dirs=case_dirs,
    T=2,
    transform=None,
    patch_size=(128, 128, 64),
    num_classes=4,
    debug=True  # 启用调试，能及时发现标签超界问题
)

sample = train_dataset[0]
x_seq, label = sample  # 解包元组

print(f"x_seq shape: {x_seq.shape}")
print(f"label shape: {label.shape}")