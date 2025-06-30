import torch
import numpy as np
from metrics import dice_score_braTS, compute_hd95
import time
from inference_helper import TemporalSlidingWindowInference
from config import config as cfg
from collections import defaultdict


class SpikeRateMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.spike_records = defaultdict(list)  # 每层的每step spike rate
        self.handles = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            # output: expected shape = [T, B, C, D, H, W]
            if isinstance(output, torch.Tensor) and output.dim() == 6:
                # Time axis is 0
                T, B, C, D, H, W = output.shape
                spike_per_t = output.float().reshape(T, -1).mean(dim=1)  # [T], 每帧平均激活
                self.spike_records[name].append(spike_per_t.cpu())  # append [T] tensor
        return hook

    def register_hooks(self, model):
        self.reset()
        for name, module in model.named_modules():
            # 可根据你模型的结构筛选 spiking 层
            if isinstance(module, torch.nn.Module) and 'Spiking' in str(type(module)):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def get_layer_avg_rates(self):
        # 对每层返回：平均 spike rate（跨时间 + 跨 step）
        return {
            name: torch.cat(records, dim=0).mean().item()
            for name, records in self.spike_records.items()
        }

    def get_full_record(self):
        # 返回每层每 step 所有时间帧的记录
        return {
            name: torch.stack(records)  # shape: [N_steps, T]
            for name, records in self.spike_records.items()
        }

    def print_summary(self):
        print("\n Spike Rate Summary:")
        for k, v in self.get_layer_avg_rates().items():
            print(f"  {k}: {v:.4f}")



# 训练配置
def get_scheduler(
    optimizer,
    num_warmup_epochs,
    num_total_epochs,
    base_lr,
    min_lr=1e-6,
    scheduler='cosine',
    power=2.0):

    if scheduler == 'cosine':
        # 余弦退火 + 线性warmup
        def lr_lambda(epoch):
            if epoch < num_warmup_epochs:
                return float(epoch) / float(max(1, num_warmup_epochs))
            else:
                progress = float(epoch - num_warmup_epochs) / float(max(1, num_total_epochs - num_warmup_epochs))
                cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
                decayed = (1 - min_lr / base_lr) * cosine_decay + min_lr / base_lr
                return decayed.item()
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler == 'polynomial':
        # 多项式衰减（无warmup）
        def lr_lambda(epoch):
            if epoch < num_warmup_epochs:
                return float(epoch) / float(max(1, num_warmup_epochs))
            else:
                progress = float(epoch) / float(max(1, num_total_epochs))
                poly_decay = (1 - progress) ** power
                decayed = (1 - min_lr / base_lr) * poly_decay + min_lr / base_lr
                return decayed
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")
    
    
    
class EarlyStopping:
    def __init__(self, patience=cfg.early_stop_patience, delta=0.0, start_epoch=200):
        self.patience = patience
        self.delta = delta
        self.start_epoch = start_epoch
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, current_epoch):
        if current_epoch < self.start_epoch:
            # 还没达到起始 epoch，跳过早停判断
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = current_epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = current_epoch
            self.counter = 0



# monitor_train = SpikeRateMonitor()
# monitor_val = SpikeRateMonitor()
monitor_train = None
monitor_val = None


# 训练和验证函数
def train(train_loader, model, optimizer, criterion, device, monitor=None, debug=False, compute_time=False):
    model.train()
    if monitor:
        monitor.register_hooks(model)
        monitor.reset()
    
    running_loss = 0.0
    print('Train -------------->>>>>>>')
    for x_seq, y in train_loader:
        if compute_time:
            start_time = time.time()
        if debug:
            # 数据检查：检查输入 x_seq 和标签 y 是否包含 NaN 或 Inf
            if torch.isnan(x_seq).any() or torch.isinf(x_seq).any():
                print(f"[FATAL] x_seq contains NaN/Inf at batch, stopping.")
                break
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"[FATAL] y contains NaN/Inf at batch, stopping.")
                break
        
        x_seq = x_seq.permute(1, 0, 2, 3, 4, 5).contiguous().to(device)   # [B, T, 1, D, H, W] → [T, B, 1, D, H, W]
        if compute_time:
            time1 = time.time()
            print(f"[DEBUG] Permute time: {time1 - start_time:.4f} seconds")
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x_seq)
        
        if compute_time:
            time2 = time.time()
            print(f"[DEBUG] Model forward time: {time2 - time1:.4f} seconds")

        if debug:
            # 检查模型输出是否为 NaN 或 Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"[FATAL] model output NaN/Inf at batch, stopping.")
                print(f"Output: {output}")  # 输出模型输出，检查其值
                break
        
        if compute_time:    
            time3 = time.time()
            print(f"[DEBUG] Model output time: {time3 - time2:.4f} seconds") 
            print(f"pred device: {output.device}, target device: {y.device}, weights device: {criterion.weights.device}") 
                     
        loss = criterion(output, y)
        
        if compute_time:
            time4 = time.time()
            print(f"[DEBUG] Loss computation time: {time4 - time3:.4f} seconds")
        
        if debug:
            # 检查 loss 是否为 NaN
            if torch.isnan(loss):
                print(f"[FATAL] loss NaN at batch, stopping at epoch") 
                break
        
        if compute_time:
            time5 = time.time()

        loss.backward()
        
        if compute_time:
            time6 = time.time()
            print(f"[DEBUG] Backward time: {time6 - time5:.4f} seconds")
            
        optimizer.step()
        
        if compute_time:
            time7 = time.time()
            print(f"[DEBUG] Optimizer step time: {time7 - time6:.4f} seconds")
            
        running_loss += loss.item()
        
        if compute_time:
            end_time = time.time()
            print(f"[DEBUG] add loss time: {end_time - time7:.4f} seconds")
            print(f"[DEBUG] Batch time: {end_time - start_time:.4f} seconds")
    
    if monitor:
        monitor.print_summary()
        monitor.remove_hooks()
        
    avg_loss = running_loss / len(train_loader)
    return avg_loss



val_inferencer = None
# 初始化滑动窗口推理器
if cfg.val_crop_mode == 'sliding_window':
    val_inferencer = TemporalSlidingWindowInference(
        patch_size=cfg.patch_size,
        overlap=cfg.overlap,
        sw_batch_size=1,
        encode_method=cfg.encode_method,
        T=cfg.T,
        num_classes=cfg.num_classes
    )


def validate(val_loader, model, criterion, device, compute_hd, monitor=None, debug=False):
    model.eval()
    if monitor:
        monitor.register_hooks(model)
        monitor.reset()
    
    total_loss = 0.0
    total_dice = {'TC': 0.0, 'WT': 0.0, 'ET': 0.0}
    hd95s = []
    print('Valid -------------->>>>>>>')
    with torch.no_grad():
        for i, (x_seq, y) in enumerate(val_loader):
            if debug:
                # 数据检查：检查输入 x_seq 和标签 y 是否包含 NaN 或 Inf
                if torch.isnan(x_seq).any() or torch.isinf(x_seq).any():
                    print(f"[FATAL] x_seq contains NaN/Inf at batch, stopping.")
                    break
                if torch.isnan(y).any() or torch.isinf(y).any():
                    print(f"[FATAL] y contains NaN/Inf at batch, stopping.")
                    break
            
            x_seq = x_seq.permute(1, 0, 2, 3, 4, 5).contiguous().to(device)  # [T, B, 1, D, H, W]
            y_onehot = y.float().to(device)
            if val_inferencer:
                output = val_inferencer(x_seq, model)
            else:
                output = model(x_seq)  # [B, C, D, H, W]，未过 softmax

            loss = criterion(output, y_onehot)
            
            if debug:
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
            
    if monitor:
        monitor.print_summary()
        monitor.remove_hooks()
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_dice = {k: v / num_batches for k, v in total_dice.items()}

    if compute_hd:
        avg_hd95 = np.nanmean(hd95s)
    else:
        avg_hd95 = np.nan
    return avg_loss, avg_dice, avg_hd95


def train_one_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold, compute_hd, scheduler=None, early_stopping=None):
    train_losses = []
    val_losses = []
    val_dices = []
    val_mean_dices = []
    val_hd95s = []
    lr_history = []

    best_dice = 0.0
    min_dice_threshold = 0.65
    warmup_epochs = cfg.num_warmup_epochs
    train_crop_mode = cfg.train_crop_mode


    for epoch in range(num_epochs):
        print(f'----------[Fold {fold}] Epoch {epoch+1}/{num_epochs} ----------')
        if train_crop_mode == 'warmup_weighted_random':
        # 计算当前中心 crop 概率（线性衰减）
            if epoch < warmup_epochs:
                prob = 1.0 - epoch / warmup_epochs  # 从1.0线性下降到0.0
            else:
                prob = 0.0

            if hasattr(train_loader.dataset, 'center_crop_prob'):
                train_loader.dataset.center_crop_prob = prob
                if prob > 0:
                    print(f"Epoch {epoch+1}: center crop prob = {prob:.2f}")
            
        train_start_time = time.time()
        
        train_loss = train(train_loader, model, optimizer, criterion, device, monitor=monitor_train)
        
        # 计时结束
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f"[Fold {fold}] Epoch {epoch+1} training time: {train_elapsed_time:.2f} seconds")
        
        train_losses.append(train_loss)
        
        val_start_time = time.time()
        val_loss, val_dice, val_hd95 = validate(val_loader, model, criterion, device, compute_hd, monitor=monitor_val)
        # 计时结束
        val_end_time = time.time()
        val_elapsed_time = val_end_time - val_start_time
        print(f"[Fold {fold}] Epoch {epoch+1} val time: {val_elapsed_time:.2f} seconds")
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
            # 打印当前学习率
            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            print(f"Epoch {epoch+1} learning rate(s): {current_lrs[0]}")
            lr_history.append(current_lrs[0]) 
            
        if early_stopping is not None:
            early_stopping(val_loss, epoch+1)
            if early_stopping.early_stop:
                print(f"[Fold {fold}] Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history


# 折训练函数
def train_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold, compute_hd, scheduler, early_stopping):
    print(f"\n[Fold {fold+1}] Training Started")
    
    train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history = train_one_fold(
        train_loader, val_loader, model, optimizer, criterion, device, num_epochs, fold+1, compute_hd, scheduler, early_stopping
    )
    
    print(f"[Fold {fold+1}] Training Completed")
    
    return train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history

