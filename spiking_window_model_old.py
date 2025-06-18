import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate




# 模型结构
class SpikingPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(4, 4, 4)):
        super().__init__()
        # 使用3D卷积做patch切分，步长和kernel大小相同，实现不重叠patch划分
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.bn = nn.BatchNorm3d(embed_dim)  # 归一化层
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())  # LIF神经元作为激活

    def forward(self, x):
        x = self.proj(x)   # 卷积切分patch
        x = self.bn(x)     # 归一化
        x = self.sn(x)     # LIF脉冲激活
        return x


class SpikingSwinAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(4, 4, 4), dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # 多头自注意力，batch_first=True使输入形状为(batch, seq_len, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(embed_dim)  # 层归一化
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())  # LIF神经元激活
        self.dropout = nn.Dropout(dropout)  # dropout提升泛化

    def forward(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        
        # 输入大小必须能被窗口大小整除，否则分窗口会错位
        assert D % wd == 0 and H % wh == 0 and W % ww == 0, "输入尺寸必须能被窗口大小整除。"

        # 先把输入切成非重叠窗口，形状变为 [B, D//wd, H//wh, W//ww, C, wd, wh, ww]
        x_windows = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
        # 调整维度方便做注意力，排列成 [B, D//wd, H//wh, W//ww, C, wd, wh, ww]
        x_windows = x_windows.permute(0, 2, 4, 6, 1, 3, 5, 7)
        # 展开窗口内的tokens，转成多窗口batch格式，形状为 [B*num_windows, tokens, C]
        x_windows = x_windows.contiguous().view(-1, C, wd * wh * ww).transpose(1, 2)

        residual = x_windows  # 残差连接备份
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)  # 自注意力
        attn_out = self.dropout(attn_out)  # dropout
        x_windows = self.norm(attn_out + residual)  # 残差后层归一化
        x_windows = self.sn(x_windows)  # 脉冲激活

        # 反向重组窗口到原始空间
        x_windows = x_windows.transpose(1, 2).view(-1, C, wd, wh, ww)
        x_windows = x_windows.view(B, D // wd, H // wh, W // ww, C, wd, wh, ww)
        x_windows = x_windows.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x_reconstructed = x_windows.view(B, C, D, H, W)

        return x_reconstructed


class SpikingTransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(4, 4, 4), dropout=0.1):
        super().__init__()
        # 窗口自注意力层
        self.attn = SpikingSwinAttention(embed_dim, num_heads, window_size, dropout)
        # MLP用1x1卷积实现，带归一化和LIF激活，最后加dropout
        self.mlp = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.attn(x)  # 注意力模块
        x = self.mlp(x)   # MLP模块
        return x


class SpikingTransformer3D(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64, num_heads=4, T=2, window_size=(4, 4, 4), dropout=0.1):
        super().__init__()
        # patch切分embedding
        self.patch_embed = SpikingPatchEmbedding(in_channels, embed_dim, patch_size=window_size)
        # 两层Transformer编码块
        self.encoder_block1 = SpikingTransformerBlock3D(embed_dim, num_heads, window_size, dropout)
        self.encoder_block2 = SpikingTransformerBlock3D(embed_dim, num_heads, window_size, dropout)
        
        # 解码器，用转置卷积上采样，同时有LIF激活和dropout
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim * 2, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout),
        )
        # 输出层，通道映射到标签类别数4
        self.readout = nn.Conv3d(32, 4, kernel_size=1)
        self.T = T  # 时间步数，做时间平均

    def forward(self, x_seq):
        functional.reset_net(self)  # 每次前向重置脉冲网络状态
        spike_sum = 0
        for t in range(self.T):
            x = self.patch_embed(x_seq[t])   # patch embedding
            x1 = self.encoder_block1(x)      # 第1编码层输出
            x2 = self.encoder_block2(x1)     # 第2编码层输出

            # 跳跃连接concat
            x_skip = torch.cat([x2, x1], dim=1)

            x = self.decoder(x_skip)          # 解码器上采样
            spike_sum += self.readout(x)     # 统计多时间步输出
        return spike_sum / self.T            # 平均时间步脉冲输出