import torch
import torch.nn as nn
import numpy as np
from medpy.metric import binary
from surface_distance import compute_surface_distances, compute_robust_hausdorff



# 评估函数
def dice_score_per_class(pred, target, num_classes=4, eps=1e-5):
    """
    Compute per-class Dice and mean Dice (excluding background).
    
    Args:
        pred (Tensor): [B, D, H, W], predicted class indices
        target (Tensor): [B, D, H, W], ground truth class indices
        num_classes (int): total number of classes
        eps (float): smoothing term

    Returns:
        mean_dice (float): average Dice score over foreground classes (1, 2, 3)
        per_class_dice (list): Dice score for each foreground class
    """
    pred_onehot = torch.nn.functional.one_hot(pred, num_classes=num_classes)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes)

    # Convert to [B, C, D, H, W]
    pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

    dice_per_class = []
    for c in range(1, num_classes):  # Skip background
        p = pred_onehot[:, c]
        t = target_onehot[:, c]
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * inter + eps) / (union + eps)
        dice_per_class.append(dice)

    mean_dice = torch.mean(torch.stack(dice_per_class)).item()
    per_class_dice = [d.item() for d in dice_per_class]

    return mean_dice, per_class_dice



def dice_score_braTS(pred, target, eps=1e-5):
    """
    Compute Dice scores for TC, WT, ET masks.

    Args:
        pred (Tensor): [B, 3, D, H, W], predicted probabilities or binary masks
        target (Tensor): [B, 3, D, H, W], one-hot ground truth masks
        eps (float): smoothing term

    Returns:
        dice_dict (dict): Dice scores for TC, WT, ET
    """
    def compute_dice(p, t):
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        return (2. * inter + eps) / (union + eps)

    pred_prob = torch.sigmoid(pred)
    pred_bin = (pred_prob > 0.5).float()
    target_bin = target.float()

    dice_dict = {
        'TC': compute_dice(pred_bin[:, 0], target_bin[:, 0]).item(),
        'WT': compute_dice(pred_bin[:, 1], target_bin[:, 1]).item(),
        'ET': compute_dice(pred_bin[:, 2], target_bin[:, 2]).item()
    }

    return dice_dict




class BratsDiceMetric(nn.Module):
    def __init__(self, include_background=False, reduction='mean_channel', smooth_nr=1e-5, smooth_dr=1e-5):
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

    def compute_dice(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        if pred.dtype == torch.float32:
            pred = pred.sigmoid()

        B, C, D, H, W = pred.shape

        if not self.include_background and C > 3:
            pred = pred[:, 1:, ...]
            target = target[:, 1:, ...]
            C -= 1

        dims = (0, 2, 3, 4)
        intersection = torch.sum(pred * target, dim=dims)
        cardinality = torch.sum(pred + target, dim=dims)
        dice = (2. * intersection + self.smooth_nr) / (cardinality + self.smooth_dr)

        return dice.detach().cpu()  # [C]

    def aggregate(self, dice_list):
        if len(dice_list) == 0:
            raise RuntimeError("No data to aggregate. Provide dice_list.")

        dice_all = torch.stack(dice_list, dim=0)  # [N, C]

        if self.reduction == 'mean':
            return dice_all.mean().item()
        elif self.reduction == 'mean_channel':
            return dice_all.mean(dim=0).tolist()
        elif self.reduction == 'none':
            return dice_all.tolist()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def to_dict(self, dice_list):
        metric_names = ["WT", "TC", "ET"]
        dice_vals = self.aggregate(dice_list)
        if not isinstance(dice_vals, list):
            raise ValueError("Dice list must be a list. Use 'mean_channel' or 'none' as reduction.")
        return {k: v for k, v in zip(metric_names, dice_vals)}

        


def compute_hd95(pred, target, num_classes=4, ignore_index=0, mode='no_compute'):
    if mode == 'slow':
        return compute_hd95_slow(pred, target, num_classes, ignore_index)
    elif mode == 'fast':
        return compute_hd95_fast(pred, target, num_classes, ignore_index)
    elif mode == 'no_compute':
        return np.nan
    else:
        print(f'hd 95 mode ERROR, {mode} is not a valid mode')


def compute_hd95_slow(pred, target, num_classes=4, ignore_index=0):
    """
    计算多类 segmentation 的平均 HD95，忽略 background
    pred, target: [B, D, H, W]，值为 0/1/2/3
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    hd95s = []
    for cls in range(1, num_classes):  # 忽略 background（class 0）
        pred_bin = (pred == cls).astype(np.uint8)
        target_bin = (target == cls).astype(np.uint8)

        if np.sum(pred_bin) == 0 or np.sum(target_bin) == 0:
            continue  # 忽略无法比较的类
        try:
            hd = binary.hd95(pred_bin, target_bin)
            hd95s.append(hd)
        except:
            continue

    if len(hd95s) == 0:
        return np.nan
    return np.mean(hd95s)


def compute_hd95_fast(pred, target, spacing=(1.0, 1.0, 1.0), num_classes=4, ignore_index=0):
    """
    高效计算 3D HD95，使用 surface-distance 库。
    标签应为 0~num_classes-1，其中 ignore_index 是背景类。
    """
    pred = pred.cpu().numpy() if hasattr(pred, "cpu") else pred
    target = target.cpu().numpy() if hasattr(target, "cpu") else target

    hd95s = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_bin = (pred == cls)
        target_bin = (target == cls)

        if not np.any(pred_bin) or not np.any(target_bin):
            print(f"[HD95 Skip] Class {cls} is empty in pred or target.")
            continue

        try:
            surface_distances = compute_surface_distances(
                target_bin, pred_bin, spacing=spacing
            )
            hd95 = compute_robust_hausdorff(surface_distances, percentile=95)
            hd95s.append(hd95)
        except Exception as e:
            print(f"[HD95 Warning] Class {cls}: {e}")
            continue

    return float(np.mean(hd95s)) if hd95s else np.nan

