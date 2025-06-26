import os
os.chdir(os.path.dirname(__file__))
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from spiking_swin_unet_model import SpikingSwinUNet3D
from losses import BratsDiceLoss, BratsFocalLoss
from utils import init_weights, save_metrics_to_file
from train import train_fold, get_scheduler, EarlyStopping
from plot import plot_metrics
from data_loader import get_data_loaders
from config import config as cfg
from glob import glob
import time
import random
import numpy as np
from monai.utils import set_determinism
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set_determinism(seed)

# 主执行流程：5折交叉验证
def main():
    # 设置随机种子
    setseed(cfg.seed)
    
    # case_dirs = [os.path.join(cfg.root_dir, d) for d in os.listdir(cfg.root_dir) if os.path.isdir(os.path.join(cfg.root_dir, d))]
    case_dirs = []
    for root in cfg.root_dirs:  # e.g., ['./data/HGG', './data/LGG']
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory '{root}' does not exist or is not a directory.")
        case_dirs += sorted(glob(os.path.join(root, '*')))
        
    
    # 打印配置名
    print(cfg.device)

    # 设置损失函数和优化器
    if cfg.loss_function == 'focal':
        criterion = BratsFocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction='mean'
        ).to(cfg.device)
    elif cfg.loss_function == 'dice':
        criterion = BratsDiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            sigmoid=True,
            weights=cfg.loss_weights
        ).to(cfg.device)
    else:
        raise ValueError(f"Unsupported loss function: {cfg.loss_function}")
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    print("weights device:", criterion.weights.device)
    # 开始交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(case_dirs)):
        model = SpikingSwinUNet3D(
            num_classes=cfg.num_classes,
            window_size=cfg.window_size,
            T=cfg.T,
            step_mode=cfg.step_mode).to(cfg.device)  # 模型
        model.apply(init_weights)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.base_lr, eps=1e-8, weight_decay=1e-4)
        scheduler = get_scheduler(optimizer, cfg.num_warmup_epochs, cfg.num_epochs, 
                                  cfg.base_lr, cfg.min_lr, cfg.scheduler, cfg.power)
        early_stopping = EarlyStopping(patience=cfg.early_stop_patience, delta=0.0001)

        # 根据交叉验证划分数据集
        train_case_dirs = [case_dirs[i] for i in train_idx]
        val_case_dirs = [case_dirs[i] for i in val_idx]

        # 训练和验证数据加载器
        train_loader, val_loader = get_data_loaders(
            train_case_dirs, val_case_dirs, cfg.patch_size, cfg.batch_size, cfg.T, cfg.encode_method, cfg.num_workers
            )


        # 调用训练函数
        train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history = train_fold(
            train_loader, val_loader, model, optimizer, criterion, cfg.device, cfg.num_epochs, \
                fold, cfg.compute_hd, scheduler, early_stopping
        )
        
        # 保存指标
        save_metrics_to_file(train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history, fold)

        # 绘制训练过程的图形
        plot_metrics(
            train_losses, val_losses,  val_dices, val_mean_dices, val_hd95s, lr_history, fold
        )

    print("\nTraining and Validation completed across all folds.")

if __name__ == "__main__":
    main()