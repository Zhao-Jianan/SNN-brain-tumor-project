import torch
import numpy as np
from metrics import dice_score_braTS, compute_hd95
import time

# 训练配置
# 线性预热 + 余弦退火
def get_scheduler_with_warmup(optimizer, num_warmup_epochs, num_total_epochs, base_lr, min_lr=1e-6):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # 线性预热：从0逐渐升到base_lr
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # 余弦退火，从base_lr降到min_lr
            progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_total_epochs - num_warmup_epochs))
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
            decayed = (1 - min_lr / base_lr) * cosine_decay + min_lr / base_lr
            return decayed.item()
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 训练和验证函数
def train(train_loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    print('Train -------------->>>>>>>')
    for x_seq, y in train_loader:
        # 数据检查：检查输入 x_seq 和标签 y 是否包含 NaN 或 Inf
        if torch.isnan(x_seq).any() or torch.isinf(x_seq).any():
            print(f"[FATAL] x_seq contains NaN/Inf at batch, stopping.")
            break
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[FATAL] y contains NaN/Inf at batch, stopping.")
            break

        x_seq = x_seq.permute(1, 0, 2, 3, 4, 5).to(device)  # [B, T, 1, D, H, W] → [T, B, 1, D, H, W]
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x_seq)

        # 检查模型输出是否为 NaN 或 Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"[FATAL] model output NaN/Inf at batch, stopping.")
            print(f"Output: {output}")  # 输出模型输出，检查其值
            break

        loss = criterion(output, y)

        # 检查 loss 是否为 NaN
        if torch.isnan(loss):
            print(f"[FATAL] loss NaN at batch, stopping at epoch") 
            break

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(val_loader, model, criterion, device, compute_hd):
    model.eval()
    total_loss = 0.0
    total_dice = {'TC': 0.0, 'WT': 0.0, 'ET': 0.0}
    hd95s = []
    print('Valid -------------->>>>>>>')
    with torch.no_grad():
        for i, (x_seq, y) in enumerate(val_loader):
             # 数据检查：检查输入 x_seq 和标签 y 是否包含 NaN 或 Inf
            if torch.isnan(x_seq).any() or torch.isinf(x_seq).any():
                print(f"[FATAL] x_seq contains NaN/Inf at batch, stopping.")
                break
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"[FATAL] y contains NaN/Inf at batch, stopping.")
                break

            x_seq = x_seq.permute(1, 0, 2, 3, 4, 5).to(device)  # [T, B, 1, D, H, W]
            y_onehot = y.float().to(device)
            output = model(x_seq)  # [B, C, D, H, W]，未过 softmax

            loss = criterion(output, y_onehot)

            # 检查 output 是否为 NaN 或 Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"[FATAL] model output NaN/Inf at batch, stopping.")
                print(f"Output: {output}")
                break

            dice = dice_score_braTS(output, y_onehot)  # dict: {'TC':..., 'WT':..., 'ET':...}
            # 累加各类别的dice值
            for key in total_dice.keys():
                total_dice[key] += dice[key]

            if compute_hd:
                hd95 = compute_hd95(output, y_onehot)
                # if np.isnan(hd95):
                #     print(f"[Warning] NaN in HD95")
                hd95s.append(hd95)

            total_loss += loss.item()

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_dice = {k: v / num_batches for k, v in total_dice.items()}

    if compute_hd:
        avg_hd95 = np.nanmean(hd95s)
    else:
        avg_hd95 = np.nan
    return avg_loss, avg_dice, avg_hd95


def train_one_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold, compute_hd, scheduler=None):
    train_losses = []
    val_losses = []
    val_dices = []
    val_mean_dices = []
    val_hd95s = []

    best_dice = 0.0
    min_dice_threshold = 0.6

    for epoch in range(num_epochs):
        print(f'----------[Fold {fold}] Epoch {epoch+1}/{num_epochs} ----------')
        
        start_time = time.time()
        
        train_loss = train(train_loader, model, optimizer, criterion, device)
        
        # 计时结束
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[Fold {fold}] Epoch {epoch+1} training time: {elapsed_time:.2f} seconds")
        
        train_losses.append(train_loss)
         
        val_loss, val_dice, val_hd95 = validate(val_loader, model, criterion, device, compute_hd)
        val_mean_dice = sum(val_dice.values()) / 3
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_mean_dices.append(val_mean_dice)
        if compute_hd:
            val_hd95s.append(val_hd95)

        val_dice_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_dice.items()])



        print(f"[Fold {fold}] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Dice: {val_dice_str} | Mean: {val_mean_dice:.4f}")
        if compute_hd:
            print(f"95HD: {val_hd95:.4f}")
        
        # 保存检查点
        if val_mean_dice > best_dice and val_mean_dice >= min_dice_threshold:
            best_dice = val_mean_dice
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            print(f"[Fold {fold}] Epoch {epoch+1}: New best Dice = {val_mean_dice:.4f}, model saved.")

        if scheduler is not None:
            scheduler.step()  # 更新学习率

    return train_losses, val_losses, val_dices, val_mean_dices, val_hd95s


# 折训练函数
def train_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold, compute_hd, scheduler):
    print(f"\n[Fold {fold+1}] Training Started")
    
    train_losses, val_losses, val_dices, val_mean_dices, val_hd95s = train_one_fold(
        train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold+1, compute_hd, scheduler
    )

    print(f"[Fold {fold+1}] Training Completed")
    
    return train_losses, val_losses, val_dices, val_mean_dices, val_hd95s

