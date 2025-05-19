import torch
import numpy as np
from utils import downsample_label
from metrics import dice_score, compute_hd95


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
        
        target_down = downsample_label(y, output.shape[2:])
        loss = criterion(output, target_down)

        # 检查 loss 是否为 NaN
        if torch.isnan(loss):
            print(f"[FATAL] loss NaN at batch, stopping at epoch") 
            break
        # else:
        #     print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(val_loader, model, criterion, device, compute_hd):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    hd95s = []
    print('Valid -------------->>>>>>>')
    with torch.no_grad():
        for x_seq, y in val_loader:
             # 数据检查：检查输入 x_seq 和标签 y 是否包含 NaN 或 Inf
            if torch.isnan(x_seq).any() or torch.isinf(x_seq).any():
                print(f"[FATAL] x_seq contains NaN/Inf at batch, stopping.")
                break
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"[FATAL] y contains NaN/Inf at batch, stopping.")
                break

            x_seq = x_seq.permute(1, 0, 2, 3, 4, 5).to(device)  # [T, B, 1, D, H, W]
            y = y.squeeze(1).long().to(device)  # [B, D, H, W]

            output = model(x_seq)  # [B, 4, D, H, W]，未过 softmax

            target_down = downsample_label(y, output.shape[2:])
            loss = criterion(output, target_down)

            # 检查 output 是否为 NaN 或 Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"[FATAL] model output NaN/Inf at batch, stopping.")
                print(f"Output: {output}")
                break
            # else:
            #     print(f"Loss: {loss.item()}")

            pred = torch.argmax(output, dim=1)  # [B, D, H, W]

            dice = dice_score(pred, target_down)

            if compute_hd:
                hd95 = compute_hd95(pred, target_down)
                # if np.isnan(hd95):
                #     print(f"[Warning] NaN in HD95")
                hd95s.append(hd95)

            total_loss += loss.item()
            total_dice += dice

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    if compute_hd:
        avg_hd95 = np.nanmean(hd95s)
    else:
        avg_hd95 = np.nan
    return avg_loss, avg_dice, avg_hd95


def train_one_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold, compute_hd, scheduler=None):
    train_losses = []
    val_losses = []
    val_dices = []
    val_hd95s = []

    best_dice = 0.0
    min_dice_threshold = 0.6

    for epoch in range(num_epochs):
        print(f'----------[Fold {fold}] Epoch {epoch+1}/{num_epochs} ----------')
        train_loss = train(train_loader, model, optimizer, criterion, device)
        train_losses.append(train_loss)
         
        val_loss, val_dice, val_hd95 = validate(val_loader, model, criterion, device, compute_hd)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        if compute_hd:
            val_hd95s.append(val_hd95)

        print(f"[Fold {fold}] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | 95HD: {val_hd95:.4f}")
        
        # 保存检查点
        if val_dice > best_dice and val_dice >= min_dice_threshold:
            best_dice = val_dice
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            print(f"[Fold {fold}] Epoch {epoch+1}: New best Dice = {val_dice:.4f}, model saved.")

        if scheduler is not None:
            scheduler.step()  # 更新学习率

    return train_losses, val_losses, val_dices, val_hd95s


# 折训练函数
def train_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold, compute_hd, scheduler):
    print(f"\n[Fold {fold+1}] Training Started")
    
    train_losses, val_losses, val_dices, val_hd95s = train_one_fold(
        train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold+1, compute_hd, scheduler
    )

    print(f"[Fold {fold+1}] Training Completed")
    
    return train_losses, val_losses, val_dices, val_hd95s

