import torch
import numpy as np
from medpy.metric import binary
from surface_distance import compute_surface_distances, compute_robust_hausdorff



# 评估函数
def dice_score(pred, target, num_classes=4, eps=1e-5):
    """
    pred, target: [B, D, H, W], int class labels
    """
    pred_onehot = torch.nn.functional.one_hot(pred, num_classes=num_classes)  # [B, D, H, W, C]
    target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes)

    # 转置成 [B, C, D, H, W]
    pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

    dice_per_class = []
    for c in range(1, num_classes):  # 跳过 background class=0
        p = pred_onehot[:, c]
        t = target_onehot[:, c]
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * inter + eps) / (union + eps)
        dice_per_class.append(dice)

    return torch.mean(torch.stack(dice_per_class)).item()



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

