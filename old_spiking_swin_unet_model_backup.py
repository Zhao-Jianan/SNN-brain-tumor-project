import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


# 模型结构
class SpikingPatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(2, 2, 2)):
        super().__init__()
        self.proj = layer.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.bn = nn.BatchNorm3d(embed_dim)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

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

        if shift_size is None:
            self.shift_size = tuple(ws // 2 for ws in window_size)
        else:
            self.shift_size = shift_size

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
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

        x_windows = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
        x_windows = x_windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x_windows = x_windows.view(-1, C, wd * wh * ww).transpose(1, 2)

        residual = x_windows
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        attn_out = self.dropout(attn_out)
        x_windows = self.norm(attn_out + residual)
        x_windows = self.sn(x_windows)

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
        self.attn = SpikingShiftedWindowAttention3D(embed_dim, num_heads, window_size, shift_size, dropout)

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
    def __init__(self, in_channels=4, embed_dims=[96, 192, 384], num_heads=[3, 6, 12],
                 window_size=(2,2,2), dropout=0.1, T=8):
        super().__init__()
        self.T = T

        self.patch_embed1 = SpikingPatchEmbed3D(in_channels, embed_dims[0], patch_size=(4,4,4))
        self.patch_embed2 = SpikingPatchEmbed3D(embed_dims[0], embed_dims[1], patch_size=(2,2,2))
        self.patch_embed3 = SpikingPatchEmbed3D(embed_dims[1], embed_dims[2], patch_size=(2,2,2))

        def make_stage(embed_dim, num_head):
            return nn.Sequential(
                SpikingSwinTransformerBlock3D(embed_dim, num_head, window_size, shift=False, dropout=dropout),
                SpikingSwinTransformerBlock3D(embed_dim, num_head, window_size, shift=True, dropout=dropout)
            )

        self.stage1 = make_stage(embed_dims[0], num_heads[0])
        self.stage2 = make_stage(embed_dims[1], num_heads[1])
        self.stage3 = make_stage(embed_dims[2], num_heads[2])

        self.bottleneck = nn.Sequential(
            SpikingSwinTransformerBlock3D(embed_dims[2], num_heads[2], window_size, shift=False, dropout=dropout),
            SpikingSwinTransformerBlock3D(embed_dims[2], num_heads[2], window_size, shift=True, dropout=dropout)
        )

        self.upsample3 = SpikingUpSample3D(embed_dims[2], embed_dims[1], dropout=dropout)
        self.upsample2 = SpikingUpSample3D(embed_dims[1], embed_dims[0], dropout=dropout)
        self.upsample1 = SpikingUpSample3D(embed_dims[0], embed_dims[0] // 2, dropout=dropout)
        self.upsample0 = SpikingUpSample3D(embed_dims[0] // 2, embed_dims[0] // 4, dropout=dropout)

        self.decode3 = make_stage(embed_dims[1], num_heads[1])
        self.decode2 = make_stage(embed_dims[0], num_heads[0])
        self.decode1 = make_stage(embed_dims[0] // 4, num_heads[0])

        self.reduce3 = nn.Sequential(
            layer.Conv3d(embed_dims[2], embed_dims[1], kernel_size=1),
            nn.BatchNorm3d(embed_dims[1]),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )

        self.reduce2 = nn.Sequential(
            layer.Conv3d(embed_dims[1], embed_dims[0], kernel_size=1),
            nn.BatchNorm3d(embed_dims[0]),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )
        self.reduce1 = nn.Sequential(
            layer.Conv3d(embed_dims[0], embed_dims[0] // 2, kernel_size=1),
            nn.BatchNorm3d(embed_dims[0] // 2),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Dropout(dropout)
        )

        self.readout = nn.Sequential(
            layer.Conv3d(embed_dims[0] // 4, 4, kernel_size=1),
            neuron.LIFNode()
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
            print(f"e1.shape: {e1.shape}")
            e1 = self.stage1(e1)               # e1 shape: [B, 96, 32, 32, 32]
            print(f"e1.shape after stage: {e1.shape}")
            e2 = self.patch_embed2(e1)
            e2 = self.stage2(e2)               # e2 shape: [B, 192, 16, 16, 16]
            e3 = self.patch_embed3(e2)
            e3 = self.stage3(e3)               # e3 shape: [B, 384, 8, 8, 8]

            x_bottle = self.bottleneck(e3)     # x_bottle shape: [B, 384, 8, 8, 8]

            d3 = self.upsample3(x_bottle)      # d3 shape: [B, 192, 16, 16, 16]
            d3 = torch.cat([d3, e2], dim=1)    # d3 shape after concat: [B, 384, 16, 16, 16]
            d3 = self.reduce3(d3)              # d3 shape after reduce: [B, 192, 16, 16, 16]
            d3 = self.decode3(d3)              # d3 shape after decode: [B, 192, 16, 16, 16]

            d2 = self.upsample2(d3)            # d2 shape: [B, 96, 32, 32, 32]   
            d2 = torch.cat([d2, e1], dim=1)    # d2 shape after concat: [B, 192, 32, 32, 32]
            d2 = self.reduce2(d2)              # d2 shape after reduce: [B, 96, 32, 32, 32]
            d2 = self.decode2(d2)              # d2 shape after decode: [B, 96, 32, 32, 32]

            
            d1 = self.upsample1(d2)            # d1 shape: [B, 24, 64, 64, 64]
            print(f'd1.shape:{d1.shape}')
            d1 = self.decode1(d1)              # d1 shape after decode: [B, 24, 64, 64, 64]
            print(f'd1.shape after decode:{d1.shape}')

            out_t = self.readout(d1)          # out_t shape: [B, 4, 64, 64, 64]
            print(f'out_t.shape:{out_t.shape}')
            spike_sum = out_t if spike_sum is None else spike_sum + out_t

        return spike_sum / self.T