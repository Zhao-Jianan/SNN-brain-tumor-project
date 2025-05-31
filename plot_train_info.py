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


# file_name = 'train_info3.txt'
# val_hd95s = []
# fold_number = 1
# train_losses, val_losses, val_dices = read_info(file_name)
# plot_metrics(train_losses, val_losses, val_dices, val_hd95s, fold_number)


# print("train_losses:", train_losses)
# print("val_losses:", val_losses)
# print("val_dices:", val_dices)






def plot_training_log_with_4_dice(log_path, save_path=None, show=True):
    """
    读取训练日志并绘制 Loss 和 Dice 曲线图

    参数:
    - log_path: str，日志文件路径
    - save_path: str or None，可选，若给定则保存图像到该路径
    - show: bool，是否显示图像

    返回:
    - 一个 dict，包含提取的数据：train_loss, val_loss, wt, tc, et, mean
    """
    train_loss, val_loss = [], []
    wt_dice, tc_dice, et_dice, mean_dice = [], [], [], []

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if "Train Loss" in line and "Val Loss" in line:
            parts = line.split("|")
            train_loss.append(float(parts[1].split(":")[1].strip()))
            val_loss.append(float(parts[2].split(":")[1].strip()))

        elif "Dice:" in line:
            # 使用正则提取所有 Dice 数值
            matches = re.findall(r"[\w]+:\s*([\d\.]+)", line)
            if len(matches) == 4:
                wt_dice.append(float(matches[0]))
                tc_dice.append(float(matches[1]))
                et_dice.append(float(matches[2]))
                mean_dice.append(float(matches[3]))
            else:
                print(f"[警告] Dice 行解析失败：{line}")

    # 创建图像
    epochs = list(range(1, len(train_loss) + 1))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：Loss
    axs[0].plot(epochs, train_loss, label="Train Loss", marker='o')
    axs[0].plot(epochs, val_loss, label="Val Loss", marker='o')
    axs[0].set_title("Train vs Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # 右图：Dice
    axs[1].plot(epochs, wt_dice, label="WT", marker='o')
    axs[1].plot(epochs, tc_dice, label="TC", marker='o')
    axs[1].plot(epochs, et_dice, label="ET", marker='o')
    axs[1].plot(epochs, mean_dice, label="Mean", marker='o')
    axs[1].set_title("Dice Scores")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Dice Score")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

    print('Done!')

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "wt_dice": wt_dice,
        "tc_dice": tc_dice,
        "et_dice": et_dice,
        "mean_dice": mean_dice,
    }


def main():
    plot_training_log_with_4_dice('./train_log/Spiking_Swin_Unet_epoch_80.txt', 
                                save_path='./train_log/Spiking_Swin_Unet_epoch_80.jpg',
                                show=False)
    
if __name__ == "__main__":
    main()