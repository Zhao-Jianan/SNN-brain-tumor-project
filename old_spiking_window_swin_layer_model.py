import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


# 模型结构
class SpikingPatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(2, 2, 2)):
        super().__init__()
        # 利用3D卷积将输入划分为不重叠patch，kernel和stride均为patch_size，实现空间降采样并映射到embed_dim维度
        self.proj = layer.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.bn = nn.BatchNorm3d(embed_dim)  # 批归一化稳定训练
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())  # LIF脉冲神经元激活函数

    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        x = self.sn(x)
        return x


class SpikingShiftedWindowAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(2, 2, 2), shift_size=None, dropout=0.1):
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
    def __init__(self, embed_dim, num_heads, window_size=(2, 2, 2), shift=False, dropout=0.1):
        super().__init__()
        shift_size = tuple(ws // 2 for ws in window_size) if shift else (0, 0, 0)
        self.attn = SpikingShiftedWindowAttention3D(embed_dim, num_heads, window_size, shift_size, dropout)

        # MLP用1x1x1卷积替代全连接，配合批归一化和LIF激活，实现非线性变换
        self.mlp = nn.Sequential(
            layer.Conv3d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class SpikingBottleneck3D(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = layer.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.sn1 = neuron.IFNode()
        self.conv2 = layer.Conv3d(hidden_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.sn2 = neuron.IFNode()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        s1 = self.sn1(x1)
        x2 = self.conv2(s1)
        x2 = self.bn2(x2)
        s2 = self.sn2(x2)
        return s2


class SpikingUpSample3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, dropout=0.1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm3d(out_channels),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.up(x)



class SpikingSwinUNet3D(nn.Module):
    def __init__(self, in_channels=4, embed_dims=[8, 16, 32], num_heads=[1, 2, 4],
                 window_size=(2,2,2), dropout=0.1, T=8):
        super().__init__()
        self.T = T

        # Patch Embedding层，逐层降采样，每次空间尺寸缩小2倍（patch_size=2），通道数翻倍
        # 输入: (B, in_channels=4, D, H, W)
        # 输出: (B, embed_dims[0], D/2, H/2, W/2)
        self.patch_embed1 = SpikingPatchEmbed3D(in_channels, embed_dims[0], patch_size=(2,2,2))
        # 输入: (B, embed_dims[0], D/2, H/2, W/2)
        # 输出: (B, embed_dims[1], D/4, H/4, W/4)
        self.patch_embed2 = SpikingPatchEmbed3D(embed_dims[0], embed_dims[1], patch_size=(2,2,2))
        # 输入: (B, embed_dims[1], D/4, H/4, W/4)
        # 输出: (B, embed_dims[2], D/8, H/8, W/8)
        self.patch_embed3 = SpikingPatchEmbed3D(embed_dims[1], embed_dims[2], patch_size=(2,2,2))


        # Encoder 每层两个block（一个shift，一个不shift），尺寸不变
        # 输入输出shape均为输入的shape
        self.encoder1_block1 = SpikingSwinTransformerBlock3D(embed_dims[0], num_heads[0], window_size, shift=False, dropout=dropout)
        self.encoder1_block2 = SpikingSwinTransformerBlock3D(embed_dims[0], num_heads[0], window_size, shift=True, dropout=dropout)

        self.encoder2_block1 = SpikingSwinTransformerBlock3D(embed_dims[1], num_heads[1], window_size, shift=False, dropout=dropout)
        self.encoder2_block2 = SpikingSwinTransformerBlock3D(embed_dims[1], num_heads[1], window_size, shift=True, dropout=dropout)

        self.encoder3_block1 = SpikingSwinTransformerBlock3D(embed_dims[2], num_heads[2], window_size, shift=False, dropout=dropout)
        self.encoder3_block2 = SpikingSwinTransformerBlock3D(embed_dims[2], num_heads[2], window_size, shift=True, dropout=dropout)


        # bottleneck
        self.bottleneck = SpikingBottleneck3D(in_channels=embed_dims[2], hidden_channels=embed_dims[2]//2)

        # Decoder上采样层（转置卷积），空间尺寸放大2倍，通道减半
        # (B, 32, D/8, H/8, W/8) -> (B, 16, D/4, H/4, W/4)
        self.up3to2 = SpikingUpSample3D(embed_dims[2], embed_dims[1], dropout=dropout)
        # (B, 16, D/4, H/4, W/4) -> (B, 8, D/2, H/2, W/2)
        self.up2to1 = SpikingUpSample3D(embed_dims[1], embed_dims[0], dropout=dropout)
        # (B, 8, D/2, H/2, W/2) -> (B, 4, D, H, W)
        self.up1to0 = SpikingUpSample3D(embed_dims[0], embed_dims[0]//2, dropout=dropout)

        # Decoder块（两个block，一个shift，一个不shift）
        # 输入通道是skip连接和up采样后拼接的通道数（即两倍对应encoder输出通道）
        # 输入输出shape均保持不变
        self.decoder3_block1 = SpikingSwinTransformerBlock3D(embed_dims[2], num_heads[2], window_size, shift=False, dropout=dropout)
        self.decoder3_block2 = SpikingSwinTransformerBlock3D(embed_dims[2], num_heads[2], window_size, shift=True, dropout=dropout)

        self.decoder2_block1 = SpikingSwinTransformerBlock3D(embed_dims[1], num_heads[1], window_size, shift=False, dropout=dropout)
        self.decoder2_block2 = SpikingSwinTransformerBlock3D(embed_dims[1], num_heads[1], window_size, shift=True, dropout=dropout)

        self.decoder1_block1 = SpikingSwinTransformerBlock3D(embed_dims[0]//2, num_heads[0], window_size, shift=False, dropout=dropout)
        self.decoder1_block2 = SpikingSwinTransformerBlock3D(embed_dims[0]//2, num_heads[0], window_size, shift=True, dropout=dropout)

        self.reduce_d3 = nn.Conv3d(embed_dims[2], embed_dims[1], kernel_size=1)
        self.reduce_d2 = nn.Conv3d(embed_dims[1], embed_dims[0], kernel_size=1)

        # 最终输出卷积，将通道数降为4（类别数）
        # 输入：(B, 4, D, H, W)
        # 输出：(B, 4, D, H, W)
        self.readout = nn.Conv3d(embed_dims[0]//2, 4, kernel_size=1)

    def forward(self, x_seq):
        functional.reset_net(self)  # 重置所有神经元状态

        spike_sum = None
        for t in range(self.T):
            x = x_seq[t]  # x shape: (B, 4, D, H, W)
            if x.dim() == 4:
                x = x.unsqueeze(0)  # 补充batch维度
            elif x.dim() != 5:
                raise ValueError(f"输入维度异常，期望4或5维，但得到{x.dim()}维")

            # --- Encoder ---
            # patch_embed1: (B, 4, D, H, W) -> (B, 8, D/2, H/2, W/2)
            e1 = self.patch_embed1(x)
            # 两个Transformer block，不改变shape
            # e1 shape: (B, 8, D/2, H/2, W/2)
            e1 = self.encoder1_block1(e1)
            e1 = self.encoder1_block2(e1)
            # patch_embed2: (B, 8, D/2, H/2, W/2) -> (B, 16, D/4, H/4, W/4)
            e2 = self.patch_embed2(e1)
            # e2 shape: (B, 16, D/4, H/4, W/4)
            e2 = self.encoder2_block1(e2)
            e2 = self.encoder2_block2(e2)
            # patch_embed3: (B, 16, D/4, H/4, W/4) -> (B, 32, D/8, H/8, W/8)
            e3 = self.patch_embed3(e2)
            # e3 shape: (B, 32, D/8, H/8, W/8)
            e3 = self.encoder3_block1(e3)
            e3 = self.encoder3_block2(e3)

            # --- Decoder ---
            # upsample d3: (B, 32, D/8, H/8, W/8) -> (B, 16, D/4, H/4, W/4)
            d3 = self.up3to2(e3)
            # 拼接skip connection e2，形状变为 (B, 32, D/4, H/4, W/4)
            d3 = torch.cat([d3, e2], dim=1)
            d3 = self.decoder3_block1(d3)
            d3 = self.decoder3_block2(d3)
            d3 = self.reduce_d3(d3)

            # upsample d2: (B, 16, D/4, H/4, W/4) -> (B, 8, D/2, H/2, W/2)
            d2 = self.up2to1(d3)
            # 拼接skip connection e1，形状变为 (B, 16, D/2, H/2, W/2)
            d2 = torch.cat([d2, e1], dim=1)
            d2 = self.decoder2_block1(d2)
            d2 = self.decoder2_block2(d2)
            d2 = self.reduce_d2(d2)

            # upsample d1: (B, 8, D/2, H/2, W/2) -> (B, 4, D, H, W)
            d1 = self.up1to0(d2)
            # 通道减半后的skip连接没有，直接通过decoder块
            d1 = self.decoder1_block1(d1)
            d1 = self.decoder1_block2(d1)

            # 最终输出卷积
            #out = self.readout(d1)  # (B, 4, D, H, W)
            out = d1
            if spike_sum is None:
                spike_sum = out
            else:
                spike_sum += out
        
        if spike_sum is None:
            raise ValueError("无有效时间步输入")
        out = spike_sum / self.T  # 时间步平均
        return out


    


