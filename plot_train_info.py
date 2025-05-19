import re, os
import matplotlib.pyplot as plt

def read_info(file_name):
# 读取训练日志文件
    with open(file_name, 'r') as f:
        log_text = f.read()

    # 使用正则表达式提取需要的数值
    pattern = r"Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Dice: ([\d.]+)"

    train_losses = []
    val_losses = []
    val_dices = []

    matches = re.findall(pattern, log_text)
    for train_loss, val_loss, dice in matches:
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_dices.append(float(dice))
    return train_losses, val_losses, val_dices



def plot_metrics(train_losses, val_losses, val_dices, val_hd95s, fold_number):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

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
    plt.plot(epochs, val_dices, 'g', label='Val Dice')
    plt.title("Validation Dice Score")
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
    print('plot done')
    plt.close()


file_name = 'train_info2.txt'
val_hd95s = []
fold_number = 1
train_losses, val_losses, val_dices = read_info(file_name)
# plot_metrics(train_losses, val_losses, val_dices, val_hd95s, fold_number)


print("train_losses:", train_losses)
print("val_losses:", val_losses)
print("val_dices:", val_dices)