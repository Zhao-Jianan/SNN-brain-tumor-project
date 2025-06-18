import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class LayerNorm3D(nn.Module):
    """
    Applies LayerNorm over channel-last format for 3D inputs with shape [B, C, D, H, W],
    by internally permuting to [B, D, H, W, C], applying LayerNorm, and permuting back.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: [B, C, D, H, W]
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        return x


# 模型结构
class SpikingPatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(2, 2, 2)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm3D(embed_dim)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

    def forward(self, x):
        x = self.proj(x)   # [B, embed_dim, D', H', W']
        x = self.norm(x)
        x = self.sn(x)
        return x




class SpikingShiftedWindowAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(2, 2, 2), shift_size=None, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        if shift_size is None:
            self.shift_size = tuple(ws // 2 for ws in window_size)
        else:
            self.shift_size = shift_size

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        assert D % wd == 0 and H % wh == 0 and W % ww == 0, "输入尺寸必须能被窗口大小整除。"

        do_shift = any(s > 0 for s in self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))

        # Partition into windows
        x_windows = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
        x_windows = x_windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x_windows = x_windows.view(-1, C, wd * wh * ww).transpose(1, 2)  # [num_windows*B, window_size^3, C]

        # Multi-head attention + spiking activation
        residual = x_windows
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        attn_out = self.dropout(attn_out)
        x_windows = attn_out + residual
        x_windows = self.sn(x_windows)

        # Reverse window partitioning
        x_windows = x_windows.transpose(1, 2).view(-1, C, wd, wh, ww)
        x_windows = x_windows.view(B, D // wd, H // wh, W // ww, C, wd, wh, ww)
        x_windows = x_windows.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x_reconstructed = x_windows.view(B, C, D, H, W)

        if do_shift:
            x_reconstructed = torch.roll(x_reconstructed, shifts=(sd, sh, sw), dims=(2, 3, 4))

        return x_reconstructed



class SpikingSwinTransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(2, 2, 2), shift=False, dropout=0.1):
        super().__init__()
        shift_size = tuple(ws // 2 for ws in window_size) if shift else (0, 0, 0)
        self.norm1 = LayerNorm3D(embed_dim) 
        self.attn = SpikingShiftedWindowAttention3D(embed_dim, num_heads, window_size, shift_size, dropout)

        self.norm2 = LayerNorm3D(embed_dim) 
        self.conv = layer.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual

        # Conv + Spike
        residual = x
        x = self.norm2(x)
        x = self.conv(x)
        x = x + residual 
        x = self.sn(x)
        x = self.dropout(x)
        return x


class SpikingSwinTransformerStage3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(2, 2, 2), dropout=0.1):
        super().__init__()
        self.block_no_shift = SpikingSwinTransformerBlock3D(embed_dim, num_heads, window_size, shift=False, dropout=dropout)
        self.block_shift = SpikingSwinTransformerBlock3D(embed_dim, num_heads, window_size, shift=True, dropout=dropout)

    def forward(self, x):
        x = self.block_no_shift(x)
        x = self.block_shift(x)
        return x


class SpikingPatchExpand3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2,2,2), stride=2, dropout=0.1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            LayerNorm3D(out_channels),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.up(x)


class SpikingConcatReduce3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        concat_channels = in_channels * 2
        self.norm = LayerNorm3D(concat_channels)
        self.reduce = nn.Sequential(
            nn.Conv3d(concat_channels, out_channels, kernel_size=1),
            LayerNorm3D(out_channels)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # dim=1 for channel concat
        x = self.norm(x)
        x = self.reduce(x)
        return x
    

class SpikingAddConverge3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm3D(channels)

    def forward(self, x1, x2):
        x = x1 + x2  # skip connection by addition
        x = self.norm(x)
        return x



class SpikingSwinUNet3D(nn.Module):
    def __init__(self, in_channels=4, embed_dim=96, num_heads=[3, 6, 12],
                 window_size=(2,2,2), dropout=0.1, T=8):
        super().__init__()
        self.T = T

        self.patch_embed1 = SpikingPatchEmbed3D(in_channels, embed_dim, patch_size=(4,4,4))
        self.down_stage1 = SpikingSwinTransformerStage3D(embed_dim, num_heads[0], window_size, dropout)

        self.patch_embed2 = SpikingPatchEmbed3D(embed_dim, embed_dim * 2, patch_size=(2,2,2))
        self.down_stage2 = SpikingSwinTransformerStage3D(embed_dim * 2, num_heads[1], window_size, dropout)

        self.patch_embed3 = SpikingPatchEmbed3D(embed_dim * 2, embed_dim * 4, patch_size=(2,2,2))
        self.feature_stage = SpikingSwinTransformerStage3D(embed_dim * 4, num_heads[2], window_size, dropout)


        self.patch_expand2 = SpikingPatchExpand3D(embed_dim * 4, embed_dim * 2, dropout=dropout)
        self.up_stage2 = SpikingSwinTransformerStage3D(embed_dim * 2, num_heads[1])
        self.converge2 = SpikingAddConverge3D(embed_dim * 2)

        self.patch_expand1 = SpikingPatchExpand3D(embed_dim * 2, embed_dim, dropout=dropout)
        self.up_stage1 = SpikingSwinTransformerStage3D(embed_dim, num_heads[0])
        self.converge1 = SpikingAddConverge3D(embed_dim)


        self.final_expand = SpikingPatchExpand3D(embed_dim, embed_dim // 4, kernel_size=4, stride=4, dropout=dropout)

        self.readout = nn.Sequential(
            layer.Conv3d(embed_dim // 4, 4, kernel_size=1)
        )

    def forward(self, x_seq):
        functional.reset_net(self)
        spike_sum = None

        for t in range(self.T):
            x = x_seq[t]
            if x.dim() == 4:
                x = x.unsqueeze(0)
            elif x.dim() != 5:
                raise ValueError(f"Input shape must be 4D or 5D, but got {x.dim()}D")
            
            e1 = self.patch_embed1(x)          # x shape: [B, 4, 128, 128, 128]
            e1 = self.down_stage1(e1)               # e1 shape: [B, 96, 32, 32, 32]
            e2 = self.patch_embed2(e1)
            e2 = self.down_stage2(e2)               # e2 shape: [B, 192, 16, 16, 16]

            feature = self.patch_embed3(e2)
            feature = self.feature_stage(feature)       # e3 shape: [B, 384, 8, 8, 8]

            d2 = self.patch_expand2(feature)      # d3 shape: [B, 192, 16, 16, 16]
            d2 = self.up_stage2(d2)              # d3 shape after decode: [B, 192, 16, 16, 16]
            d2 = self.converge2(d2,e2)        # d3 shape after converge: [B, 192, 16, 16, 16]

            d1 = self.patch_expand1(d2)            # d2 shape: [B, 96, 32, 32, 32] 
            d1 = self.up_stage1(d1)                 # d2 shape after decode: [B, 96, 32, 32, 32]  
            d1 = self.converge1(d1,e1)       # d2 shape after converge: [B, 96, 32, 32, 32]

            d0 = self.final_expand(d1)            # d1 shape: [B, 24, 128, 128, 128]

            out_t = self.readout(d0)          # out_t shape: [B, 4, 128, 128, 128]
            spike_sum = out_t if spike_sum is None else spike_sum + out_t

        return spike_sum / self.T