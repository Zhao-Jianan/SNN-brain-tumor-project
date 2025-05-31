import os
import matplotlib.pyplot as plt


# 绘制图形的函数
def plot_metrics(train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, fold_number):
    epochs = range(1, len(train_losses) + 1)

    # 拆分 val_dices 为多个类别
    val_dices_wt = [d['WT'] for d in val_dices]
    val_dices_tc = [d['TC'] for d in val_dices]
    val_dices_et = [d['ET'] for d in val_dices]

    plt.figure(figsize=(15, 4))

    # Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Dice 曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_dices_wt, label='WT Dice')
    plt.plot(epochs, val_dices_tc, label='TC Dice')
    plt.plot(epochs, val_dices_et, label='ET Dice')
    plt.plot(epochs, val_mean_dices, label='Mean Dice', linestyle='--', color='black')
    plt.title("Validation Dice Scores")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    # 95HD 曲线（仅当长度匹配时才绘制）
    if len(val_hd95s) == len(epochs):
        plt.subplot(1, 3, 3)
        plt.plot(epochs, val_hd95s, 'r', label='Val 95HD')
        plt.title("Validation 95% Hausdorff Distance")
        plt.xlabel("Epoch")
        plt.ylabel("95HD")
        plt.legend()
    else:
        print(f"[Warning] Skipping 95HD plot: val_hd95s has length {len(val_hd95s)}, expected {len(epochs)}")

    plt.tight_layout()

    os.makedirs("visualise", exist_ok=True)
    save_path = f"visualise/metrics_fold{fold_number}.png"
    plt.savefig(save_path)
    plt.close()