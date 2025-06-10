import torch
import torch.nn.functional as F
import torch.nn as nn
import os, json

def downsample_label(label, size):
    label = label.unsqueeze(1).float()  # [B,1,D,H,W]
    label_down = F.interpolate(label, size=size, mode='nearest')
    return label_down.squeeze(1).long()  # [B,D,H,W]


def init_weights(module):
    """
    针对使用LIFNode的网络，采用Xavier初始化Conv3d和Linear层权重，BatchNorm层权重初始化为1，偏置初始化为0。
    """
    if isinstance(module, nn.Conv3d):
        # Xavier初始化，适合LIFNode激活
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.MultiheadAttention):
        # 对于 MultiheadAttention 手动初始化其包含的 Linear 层
        nn.init.xavier_normal_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0)
        nn.init.xavier_normal_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            nn.init.constant_(module.out_proj.bias, 0)


def save_metrics_to_file(train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, fold, output_dir="metrics"):
    os.makedirs(output_dir, exist_ok=True)

    # 拆分 val_dices 字典数组为独立的列表
    val_dices_wt = [d['WT'] for d in val_dices]
    val_dices_tc = [d['TC'] for d in val_dices]
    val_dices_et = [d['ET'] for d in val_dices]

    data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_dices_wt": val_dices_wt,
        "val_dices_tc": val_dices_tc,
        "val_dices_et": val_dices_et,
        "val_mean_dices": val_mean_dices,
        "val_hd95s": val_hd95s
    }

    filepath = os.path.join(output_dir, f"fold_{fold+1}_metrics.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
