import torch
import torch.nn.functional as F

T = 8

# 假设输入x已归一化到[0,1]
x = torch.rand(1, 4, 128, 128, 128)  # [B, C, D, H, W]

print('Input x:', x.shape, x.min().item(), x.max().item())

B, C, D, H, W = x.shape

# Flatten所有元素到一维
x_flat = x.view(-1)  # [B*C*D*H*W]

# 计算t_f，线性编码
t_f = ((T - 1) * (1. - x_flat)).round().long()  # [B*C*D*H*W]

# one-hot编码：shape -> [B*C*D*H*W, T]
one_hot = F.one_hot(t_f, num_classes=T).float()

# reshape回 [B, C, D, H, W, T]
one_hot = one_hot.view(B, C, D, H, W, T)

# 调换维度，把T维放最前面 -> [T, B, C, D, H, W]
one_hot = one_hot.permute(5, 0, 1, 2, 3, 4)

print('Output spikes:', one_hot.shape)
