import os
import torch.optim as optim
from sklearn.model_selection import KFold

from spiking_swin_model import SpikingSwinTransformer3D
from losses import DiceCrossEntropyLoss
from utils import init_weights, save_metrics_to_file
from train import train_fold, get_scheduler_with_warmup
from plot import plot_metrics
from data_loader import get_data_loaders
from config import device, root_dir, T, num_epochs, batch_size, base_lr, k_folds, \
                    class_weights, num_workers, compute_hd, num_warmup_epochs, min_lr


# 主执行流程：5折交叉验证
def main():
    case_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    # 打印配置名
    print(device)

    # 设置损失函数和优化器
    criterion = DiceCrossEntropyLoss(weight=class_weights, dice_weight=2.0)
    model = SpikingSwinTransformer3D(T=T).to(device)  # 模型
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    kf = KFold(n_splits=k_folds, shuffle=True)
    scheduler = get_scheduler_with_warmup(optimizer, num_warmup_epochs, num_epochs, base_lr, min_lr)

    # 开始交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(case_dirs)):
        # 根据交叉验证划分数据集
        train_case_dirs = [case_dirs[i] for i in train_idx]
        val_case_dirs = [case_dirs[i] for i in val_idx]

        # 训练和验证数据加载器
        train_loader, val_loader = get_data_loaders(train_case_dirs, val_case_dirs, batch_size, T, num_workers)

        # 调用训练函数
        train_losses, val_losses, val_dices, val_hd95s = train_fold(
            train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold, compute_hd, scheduler
        )

        # 保存指标
        save_metrics_to_file(train_losses, val_losses, val_dices, val_hd95s, fold)

        # 绘制训练过程的图形
        plot_metrics(
            train_losses, val_losses,  val_dices, val_hd95s,fold
        )

    print("\nTraining and Validation completed across all folds.")

if __name__ == "__main__":
    main()