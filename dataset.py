import os, traceback
from typing import Union
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConvertToMultiChannelBasedOnBratsClassesd,
    MapTransform, RandFlipd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd,
    ToTensord, Orientationd, Spacingd
)
from monai.data import Dataset as MonaiDataset
from monai.transforms.utils import allow_missing_keys_mode
from spikingjelly.clock_driven.encoding import PoissonEncoder, LatencyEncoder, WeightedPhaseEncoder
from config import config as cfg
from typing import Mapping, Hashable, Sequence
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import Transform
from monai.utils import TransformBackends
import random

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
        self.num_warmup_epochs = cfg.num_warmup_epochs
        self.center_crop_prob = 1.0  # 默认100%中心crop
        self.crop_mode = cfg.crop_mode
        self.num_classes = num_classes
        self.mode = mode
        self.debug = debug
        self.encode_method = encode_method
        self.poisson_encoder = PoissonEncoder()
        self.latency_encoder = LatencyEncoder(self.T)
        self.weighted_phase_encoder = WeightedPhaseEncoder(self.T)
        
        self.sep = cfg.modality_separator
        self.suffix = cfg.image_suffix
        self.et_label = cfg.et_label
        

        # 读取数据，自动封装成 MetaTensor (带affine)
        self.load_transform = Compose([
            LoadImaged(keys=["image", "label"]),  # 加载 nii，自动带 affine
            EnsureChannelFirstd(keys=["image", "label"]),  # 保证通道维度在前 (C, D, H, W)
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),  # [0: TC, 1: WT, 2: ET]
        ])
        
        self.load_transform_custom_convert = Compose([
            LoadImaged(keys=["image", "label"]),  # 加载 nii，自动带 affine
            EnsureChannelFirstd(keys=["image", "label"]),  # 保证通道维度在前 (C, D, H, W)
            ConvertToMultiChannelBasedOnBrats2023Classesd(
                keys="label"
            ),  # [0: TC, 1: WT, 2: ET]
        ])      
        
        # 统一空间预处理
        self.preprocess = Compose([
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ])
        
        self.normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)

        # 数据增强 pipeline
        self.train_transform = Compose([
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"])
        ])

        self.val_transform = Compose([
            ToTensord(keys=["image", "label"])
        ])

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data = self.data_dicts[idx]
        case_name = os.path.basename(data["label"]).replace(f"{self.sep}seg{self.suffix}", "")

        if self.et_label == 4:
            data = self.load_transform(data)  # load nii -> MetaTensor (C, D, H, W) + affine
        elif self.et_label == 3:
            data = self.load_transform_custom_convert(data)  # load nii -> MetaTensor (C, D, H, W) + affine
        else:
            raise ValueError(f"Wrong ET Label in the config: {self.et_label}")

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


        data = self.normalize(data)
        
        if self.mode == "train":
            data = self.patch_crop(data, mode=self.crop_mode)  # 随机裁剪 patch
            data = self.train_transform(data)
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
            x_seq = self.encode_spike_input(img_rescale)

            # x_seq: (T, C, D, H, W), label: (C_label, D, H, W)
            return x_seq, label
            
            
        else:
            data = self.val_transform(data)
            img = data["image"]  # Tensor (C, D, H, W)
            label = data["label"]  # Tensor (C_label, D, H, W) 
            img = img.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # x_seq: (T, C, D, H, W), label: (C_label, D, H, W)
            return img, label



    # 随机裁剪，支持 warmup 模式
    # warmup 模式下，优先裁剪肿瘤中心区域；否则随机裁剪
    def patch_crop(self, data, mode="tumor_aware_random"):
        img = data["image"]        # (C, D, H, W)
        label = data["label"]      # (C, D, H, W), one-hot
        _, D, H, W = img.shape
        pd, ph, pw = self.patch_size
        assert D >= pd and H >= ph and W >= pw, f"Patch size {self.patch_size} too big for image {img.shape}"


        if mode == "tumor_aware_random":
        # 获取肿瘤前景 mask（任意类别）
            foreground_mask = label.sum(axis=0) > 0
            foreground_voxels = np.argwhere(foreground_mask.cpu().numpy())

            if len(foreground_voxels) > 0:
                # 从前景中随机选一个点作为 patch 中心
                center = foreground_voxels[np.random.choice(len(foreground_voxels))]
                cd, ch, cw = center

                # 随机扰动中心点（增加多样性）
                cd += np.random.randint(-pd//4, pd//4 + 1)
                ch += np.random.randint(-ph//4, ph//4 + 1)
                cw += np.random.randint(-pw//4, pw//4 + 1)

                d_start = np.clip(cd - pd // 2, 0, D - pd)
                h_start = np.clip(ch - ph // 2, 0, H - ph)
                w_start = np.clip(cw - pw // 2, 0, W - pw)

            else:
                # 无肿瘤则随机裁剪
                d_start = np.random.randint(0, D - pd + 1)
                h_start = np.random.randint(0, H - ph + 1)
                w_start = np.random.randint(0, W - pw + 1)
                
        elif mode == "warmup_weighted_random":
            if random.random() < self.center_crop_prob:
                # 偏向肿瘤区域的随机裁剪（带扰动）
                foreground_mask = label.sum(axis=0) > 0
                foreground_voxels = np.argwhere(foreground_mask.cpu().numpy())

                if len(foreground_voxels) > 0:
                    center = foreground_voxels[np.random.choice(len(foreground_voxels))]
                    cd, ch, cw = center

                    # 随机扰动中心点
                    cd += np.random.randint(-pd // 4, pd // 4 + 1)
                    ch += np.random.randint(-ph // 4, ph // 4 + 1)
                    cw += np.random.randint(-pw // 4, pw // 4 + 1)

                    d_start = np.clip(cd - pd // 2, 0, D - pd)
                    h_start = np.clip(ch - ph // 2, 0, H - ph)
                    w_start = np.clip(cw - pw // 2, 0, W - pw)

                else:
                    # 没有肿瘤 → 随机裁剪
                    d_start = np.random.randint(0, D - pd + 1)
                    h_start = np.random.randint(0, H - ph + 1)
                    w_start = np.random.randint(0, W - pw + 1)

            else:
                # 纯随机裁剪（不管是否有肿瘤）
                d_start = np.random.randint(0, D - pd + 1)
                h_start = np.random.randint(0, H - ph + 1)
                w_start = np.random.randint(0, W - pw + 1)
                
        else:
            # 纯随机裁剪
            d_start = np.random.randint(0, D - pd + 1)
            h_start = np.random.randint(0, H - ph + 1)
            w_start = np.random.randint(0, W - pw + 1)

        data["image"] = img[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        data["label"] = label[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        return data


    
    def rescale_to_unit_range(self, x: torch.Tensor) -> torch.Tensor:
        # 逐个样本 min-max 归一化，不改变整体分布，只用于编码器
        x_min = x.amin(dim=[1, 2, 3], keepdim=True)
        x_max = x.amax(dim=[1, 2, 3], keepdim=True)
        x_rescaled = (x - x_min) / (x_max - x_min + 1e-8)
        return x_rescaled.clamp(0., 1.)
    
    def encode_spike_input(self, img_rescale: torch.Tensor) -> torch.Tensor:
        """
        对归一化图像进行脉冲编码，支持 poisson / latency / weighted_phase。
        输入:
            img_rescale: torch.Tensor, shape (B, C, D, H, W), 数值应已在 [0, 1] 区间 
        """
        if self.encode_method == 'poisson':
            x_seq = torch.stack([self.poisson_encoder(img_rescale) for _ in range(self.T)], dim=0)
        elif self.encode_method == 'latency':
            img_rescale = img_rescale.unsqueeze(0)  # (1,C,D,H,W)
            self.latency_encoder.encode(img_rescale)  # (T,1,C,D,H,W)
            spike = self.latency_encoder.spike
            x_seq = spike.squeeze(1)  # (T,C,D,H,W)
        elif self.encode_method == 'weighted_phase':
            img_rescale = img_rescale * (1 - 2**(-self.T))
            img_rescale = img_rescale.unsqueeze(0)  # (1,C,D,H,W)
            self.weighted_phase_encoder.encode(img_rescale)  # (T,1,C,D,H,W)
            spike = self.weighted_phase_encoder.spike.float()
            x_seq = spike.squeeze(1)  # (T,C,D,H,W)    
        else:
            raise NotImplementedError(f"Encoding method '{self.encode_method}' is not implemented.")
        # x_seq: (T, C, D, H, W), label: (C_label, D, H, W)
        return x_seq



NdarrayOrTensor = Union[np.ndarray, torch.Tensor, MetaTensor]

class ConvertToMultiChannelBasedOnBrats2023Classes(Transform):
    """
    Convert BraTS 2023 labels (NCR/NET=1, ED=2, ET=3) to multi-channel labels:
      - TC (Tumor core): labels 1 or 3
      - WT (Whole tumor): labels 1 or 2 or 3
      - ET (Enhancing tumor): label 3
    Preserves MetaTensor meta information.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # 如果是 MetaTensor，取出数据和meta
        if isinstance(img, MetaTensor):
            meta = img.meta
            img_data = img.as_tensor()  # 返回 torch.Tensor
        elif isinstance(img, torch.Tensor):
            meta = None
            img_data = img
        else:
            meta = None
            img_data = torch.from_numpy(img)  # 统一转 torch.Tensor 方便操作

        # 如果是4维且第0维是1，去掉该通道维度（确保标签为 (D,H,W)）
        if img_data.ndim == 4 and img_data.shape[0] == 1:
            img_data = img_data.squeeze(0)

        # 生成3个通道的标签mask
        tc = (img_data == 1) | (img_data == 3)
        wt = (img_data == 1) | (img_data == 2) | (img_data == 3)
        et = (img_data == 3)

        result = torch.stack([tc, wt, et], dim=0).to(dtype=torch.uint8)

        # 如果有meta信息，用MetaTensor封装返回，否则直接返回tensor
        if meta is not None:
            return MetaTensor(result, meta=meta)
        else:
            return result
        
        
        
class ConvertToMultiChannelBasedOnBrats2023Classesd(MapTransform):
    def __init__(self, keys: Sequence[Hashable], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBrats2023Classes()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d