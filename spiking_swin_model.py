import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate


# 模型结构
class SpikingPatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(4, 4, 4)):
        super().__init__()
        # 利用3D卷积将输入划分为不重叠patch，kernel和stride均为patch_size，实现空间降采样并映射到embed_dim维度
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.bn = nn.BatchNorm3d(embed_dim)  # 批归一化稳定训练
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())  # LIF脉冲神经元激活函数

    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        x = self.sn(x)
        return x


class SpikingShiftedWindowAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(4, 4, 4), shift_size=None, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # 默认shift为窗口一半，实现滑动窗口机制；否则不滑动
        if shift_size is None:
            self.shift_size = tuple(ws // 2 for ws in window_size)
        else:
            self.shift_size = shift_size

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)  # 多头自注意力
        self.norm = nn.LayerNorm(embed_dim)  # 归一化层
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())  # 脉冲激活
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        # 输入尺寸必须是窗口大小整数倍，便于划分非重叠窗口
        assert D % wd == 0 and H % wh == 0 and W % ww == 0, "输入尺寸必须能被窗口大小整除。"

        do_shift = any(s > 0 for s in self.shift_size)
        if do_shift:
            # 滑动窗口：负向滚动以模拟窗口偏移
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))

        # 划分非重叠窗口，调整维度为 (num_windows*B, wd*wh*ww, C) 方便多头注意力
        x_windows = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
        x_windows = x_windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x_windows = x_windows.view(-1, C, wd * wh * ww).transpose(1, 2)

        residual = x_windows  # 残差连接
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        attn_out = self.dropout(attn_out)
        x_windows = self.norm(attn_out + residual)
        x_windows = self.sn(x_windows)  # LIF激活

        # 恢复窗口结构并还原原始形状
        x_windows = x_windows.transpose(1, 2).view(-1, C, wd, wh, ww)
        x_windows = x_windows.view(B, D // wd, H // wh, W // ww, C, wd, wh, ww)
        x_windows = x_windows.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x_reconstructed = x_windows.view(B, C, D, H, W)

        if do_shift:
            # 逆向滚动，恢复原始窗口位置
            x_reconstructed = torch.roll(x_reconstructed, shifts=(sd, sh, sw), dims=(2, 3, 4))

        return x_reconstructed


class SpikingSwinTransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(4, 4, 4), shift=False, dropout=0.1):
        super().__init__()
        shift_size = tuple(ws // 2 for ws in window_size) if shift else (0, 0, 0)
        self.attn = SpikingShiftedWindowAttention3D(embed_dim, num_heads, window_size, shift_size, dropout)

        # MLP用1x1x1卷积替代全连接，配合批归一化和LIF激活，实现非线性变换
        self.mlp = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class SpikingSwinTransformer3D(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64, num_heads=4, T=2, window_size=(4, 4, 4), dropout=0.1):
        super().__init__()
        # Patch embedding模块：下采样至原始的1/4 (即128→32)
        self.patch_embed = SpikingPatchEmbed3D(in_channels, embed_dim, patch_size=window_size)

        # 两个编码块交替使用滑动窗口机制，增强局部和全局感受野
        self.encoder_block1 = SpikingSwinTransformerBlock3D(embed_dim, num_heads, window_size, shift=False, dropout=dropout)
        self.encoder_block2 = SpikingSwinTransformerBlock3D(embed_dim, num_heads, window_size, shift=True, dropout=dropout)

        # 解码器第一层上采样，利用转置卷积上采样恢复空间尺寸，上采样回到64×64×64, 并用LIF激活与dropout增强泛化
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose3d(embed_dim * 2, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout),
        )

        # 解码器第二层上采样，从64×64×64到128×128×128
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),  # 64 → 128
            nn.BatchNorm3d(16),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout),
        )

        self.readout = nn.Conv3d(16, 4, kernel_size=1)  # 输出4个类别通道
        self.T = T  # 脉冲时间步数

    def forward(self, x_seq):
        functional.reset_net(self)  # 重置LIF神经元状态，防止跨时间步干扰
        spike_sum = 0
        for t in range(self.T):
            x = x_seq[t]
            # 如果缺少 batch 维度，则补一个维度，变成 batch=1
            if x.dim() == 4:  # 形状是 [C, D, H, W]
                x = x.unsqueeze(0)  # 变成 [1, C, D, H, W]
            elif x.dim() != 5:
                raise ValueError(f"输入维度异常，期望4维或5维张量，但得到{x.dim()}维")

            x = self.patch_embed(x)  # patch embedding
            x1 = self.encoder_block1(x)     # 编码器块1（无滑动）
            x2 = self.encoder_block2(x1)    # 编码器块2（滑动窗口）
            x_skip = torch.cat([x2, x1], dim=1)  # 跳跃连接融合特征 shape: [B, 128, 32, 32, 32]
            x = self.decoder_block1(x_skip)      # 解码器块1上采样 64x64x64 shape: [B, 32, 64, 64, 64]
            x = self.decoder_block2(x)           # 解码器块2上采样 128x128x128 shape: [B, 16, 128, 128, 128]
            out = self.readout(x)                # 输出层 shape: [B, 4, 128, 128, 128]
            spike_sum += out                     # 读出层累积脉冲响应
        return spike_sum / self.T  # 脉冲平均输出，提高稳定性