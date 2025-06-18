import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base
from config import config as cfg

# 使用spikingjelly的多步模式
class LayerNorm3D(base.MemoryModule):
    """
    支持单步（step_mode='s'）和多步（step_mode='m'）模式。
    """

    def __init__(self, num_channels, step_mode='s'):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        assert step_mode in ('s', 'm'), "step_mode must be 's' or 'm'"
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            # 单步输入 [B, C, D, H, W]
            x = x.permute(0, 2, 3, 4, 1)  # -> [B, D, H, W, C]
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)  # -> [B, C, D, H, W]
            return x

        elif self.step_mode == 'm':
            # 多步输入 [T, B, C, D, H, W]
            T, B, C, D, H, W = x.shape
            x = x.view(T * B, C, D, H, W)  # 合并时间和batch
            x = x.permute(0, 2, 3, 4, 1)  # -> [T*B, D, H, W, C]
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)  # -> [T*B, C, D, H, W]
            x = x.view(T, B, C, D, H, W)   # 拆回 [T, B, C, D, H, W]
            return x



# 模型结构
class SpikingPatchEmbed3D(base.MemoryModule):
    def __init__(self, in_channels, embed_dim, patch_size=(2, 2, 2), step_mode='m'):
        super().__init__()
        self.proj = layer.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, step_mode=step_mode)
        self.norm = LayerNorm3D(embed_dim, step_mode=step_mode)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
        functional.set_step_mode(self, step_mode=step_mode)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.sn(x)
        return x



class SpikingShiftedWindowAttention3D(base.MemoryModule):
    def __init__(self, embed_dim, num_heads, window_size=(2, 2, 2), shift_size=None, dropout=0.1, step_mode='s'):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.step_mode = step_mode

        if shift_size is None:
            self.shift_size = tuple(ws // 2 for ws in window_size)
        else:
            self.shift_size = shift_size

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
        self.dropout = layer.Dropout(dropout, step_mode=step_mode)

    def _window_partition(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size

        x_windows = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
        x_windows = x_windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x_windows = x_windows.view(-1, C, wd * wh * ww).transpose(1, 2)  # [num_windows*B, window_size^3, C]
        return x_windows, B, D, H, W

    def _window_reverse(self, x_windows, B, D, H, W):
        wd, wh, ww = self.window_size
        C = x_windows.size(2)

        x_windows = x_windows.transpose(1, 2).contiguous().view(-1, C, wd, wh, ww)
        x_windows = x_windows.view(B, D // wd, H // wh, W // ww, C, wd, wh, ww)
        x_windows = x_windows.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x_reconstructed = x_windows.view(B, C, D, H, W)
        return x_reconstructed

    def forward_single_step(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        assert D % wd == 0 and H % wh == 0 and W % ww == 0, "输入尺寸必须能被窗口大小整除。"

        do_shift = any(s > 0 for s in self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))

        x_windows, B, D, H, W = self._window_partition(x)

        # Attention (单步)
        residual = x_windows
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        attn_out = self.dropout(attn_out)
        x_windows = attn_out + residual
        x_windows = self.sn(x_windows)

        x_reconstructed = self._window_reverse(x_windows, B, D, H, W)

        if do_shift:
            x_reconstructed = torch.roll(x_reconstructed, shifts=(sd, sh, sw), dims=(2, 3, 4))

        return x_reconstructed

    def forward_multi_step(self, x):
        # x: [T, B, C, D, H, W]
        T, B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        assert D % wd == 0 and H % wh == 0 and W % ww == 0, "输入尺寸必须能被窗口大小整除。"

        do_shift = any(s > 0 for s in self.shift_size)
        if do_shift:
            # Shift all time steps
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(3, 4, 5))  # T=0,B=1,C=2,D=3,H=4,W=5

        # 重新合并T和B作为batch
        x = x.view(T * B, C, D, H, W)  # 合并T和B作为大batch

        # 窗口分割
        x_windows, new_B, D, H, W = self._window_partition(x)  # new_B = T*B

        # Attention
        attn_out = []
        for t in range(T):
            # 注意：x_windows 的顺序是 [t0_b0_win0, t0_b0_win1, ..., t1_b0_win0, ...]
            start = t * B * (D // wd) * (H // wh) * (W // ww)
            end = (t + 1) * B * (D // wd) * (H // wh) * (W // ww)
            x_w = x_windows[start:end]  # [B * num_windows, win_size^3, C]

            residual = x_w
            out, _ = self.attn(x_w, x_w, x_w)
            out = out + residual
            attn_out.append(out)
            
        # 拼回所有时间步
        x_windows = torch.cat(attn_out, dim=0)  # [T*B*num_win, win_size^3, C]
        
        # dropout + spike：支持多时间步，reshape 成 [T, B, ...]
        x_windows = x_windows.view(T, B, -1, self.embed_dim)  # [T, B*num_win, win_size^3, C]
        x_windows = self.dropout(x_windows)  # 支持 step_mode='m'
        x_windows = self.sn(x_windows)       # 支持 step_mode='m'
        x_windows = x_windows.view(T * B, -1, self.embed_dim)  # reshape 回原来的展开形式

        # 反向还原
        x_reconstructed = self._window_reverse(x_windows, new_B, D, H, W)

        # new_B = T * B, reshape回[T, B, C, D, H, W]
        x_reconstructed = x_reconstructed.view(T, B, C, D, H, W)

        if do_shift:
            x_reconstructed = torch.roll(x_reconstructed, shifts=(sd, sh, sw), dims=(3, 4, 5))

        return x_reconstructed

    def forward(self, x):
        if self.step_mode == 's':
            # x: [B, C, D, H, W]
            return self.forward_single_step(x)
        elif self.step_mode == 'm':
            # x: [T, B, C, D, H, W]
            return self.forward_multi_step(x)
        else:
            raise NotImplementedError(f"Unsupported step_mode: {self.step_mode}")



class SpikingSwinTransformerBlock3D(base.MemoryModule):
    def __init__(self, embed_dim, mlp_dim, num_heads, window_size=(2, 2, 2), shift=False, dropout=0.1, step_mode='s'):
        super().__init__()       
        shift_size = tuple(ws // 2 for ws in window_size) if shift else (0, 0, 0)

        self.norm1 = LayerNorm3D(embed_dim, step_mode=step_mode)
        self.attn = SpikingShiftedWindowAttention3D(embed_dim, num_heads, window_size, shift_size, dropout, step_mode=step_mode)

        self.norm2 = LayerNorm3D(embed_dim, step_mode=step_mode)
        
        self.linear1 = layer.Linear(embed_dim, mlp_dim, step_mode=step_mode)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
        self.linear2 = layer.Linear(mlp_dim, embed_dim, step_mode=step_mode)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)       
        self.dropout = layer.Dropout(dropout, step_mode=step_mode)


    def forward(self, x):

        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        
        T, B, C, D, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2)  # (T, B, D, H, W, C)
        x = x.reshape(T * B, D * H * W, C)  # (T*B, N, C) N = D*H*W

        x = self.linear1(x)
        x = self.sn1(x)
        x = self.linear2(x)
        x = self.sn2(x)
        x = self.dropout(x)
        
        # 还原维度回 (T, B, C, D, H, W)
        x = x.reshape(T, B, D, H, W, C)
        x = x.permute(0, 1, 5, 2, 3, 4)
    
        x = x + residual
        return x


class SpikingSwinTransformerStage3D(base.MemoryModule):
    def __init__(self, embed_dim, num_heads, layers, window_size=(2, 2, 2), dropout=0.1, step_mode='s'):
        super().__init__()
        block_list = []
        for i in range(layers):
            shift = (i % 2 == 1)  # 奇数 block shift=True，偶数 False
            block_list.append(
                SpikingSwinTransformerBlock3D(
                    embed_dim=embed_dim,
                    mlp_dim=embed_dim*4,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=shift,
                    dropout=dropout,
                    step_mode=step_mode
                )
            )
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):
        return self.blocks(x)



class SpikingPatchExpand3D(base.MemoryModule):
    def __init__(self, in_channels, out_channels, kernel_size=(2,2,2), stride=2, dropout=0.1, step_mode='s'):
        super().__init__()
        self.conv_transpose = layer.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, step_mode=step_mode)
        self.norm = LayerNorm3D(out_channels, step_mode=step_mode)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
        functional.set_step_mode(self, step_mode)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.sn(x)
        return x


class FinalSpikingPatchExpand3D(base.MemoryModule):
    def __init__(self, in_channels, out_channels, kernel_size=(2,2,2), stride=2, dropout=0.1, step_mode='s'):
        super().__init__()
        self.conv_transpose = layer.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, step_mode=step_mode)
        self.norm = LayerNorm3D(out_channels, step_mode=step_mode)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)

        functional.set_step_mode(self, step_mode)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.sn(x)
        return x


class SpikingConcatReduce3D(base.MemoryModule):
    def __init__(self, in_channels, out_channels, step_mode='s'):
        super().__init__()
        concat_channels = in_channels * 2

        self.norm1 = LayerNorm3D(concat_channels, step_mode=step_mode)
        self.conv = layer.Conv3d(concat_channels, out_channels, kernel_size=1, step_mode=step_mode)
        self.norm2 = LayerNorm3D(out_channels, step_mode=step_mode)

        functional.set_step_mode(self, step_mode)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension
        x = self.norm1(x)
        x = self.conv(x)
        x = self.norm2(x)
        return x

    

class SpikingAddConverge3D(base.MemoryModule):
    def __init__(self, channels, step_mode='s'):
        super().__init__()
        self.norm = LayerNorm3D(channels, step_mode=step_mode)
        
        functional.set_step_mode(self, step_mode)

    def forward(self, x1, x2):
        x = x1 + x2  # skip connection by addition
        x = self.norm(x)
        return x



class SpikingSwinUNet3D(base.MemoryModule):
    def __init__(self, in_channels=4, num_classes=3, embed_dim=cfg.embed_dim, layers=[2, 2, 4, 2], num_heads=cfg.num_heads,
                 window_size=(4,4,4), dropout=0.1, T=8, step_mode='s'):
        super().__init__()
        self.T = T
        self.step_mode = step_mode

        self.patch_embed1 = SpikingPatchEmbed3D(in_channels, embed_dim, patch_size=(4,4,4), step_mode=step_mode)
        self.down_stage1 = SpikingSwinTransformerStage3D(
            embed_dim, num_heads[0], layers[0], window_size, dropout, step_mode=step_mode)

        self.patch_embed2 = SpikingPatchEmbed3D(embed_dim, embed_dim * 2, patch_size=(2,2,2), step_mode=step_mode)
        self.down_stage2 = SpikingSwinTransformerStage3D(
            embed_dim * 2, num_heads[1], layers[1], window_size, dropout, step_mode=step_mode)

        self.patch_embed3 = SpikingPatchEmbed3D(embed_dim * 2, embed_dim * 4, patch_size=(2,2,2), step_mode=step_mode)
        self.down_stage3 = SpikingSwinTransformerStage3D(
            embed_dim * 4, num_heads[2], layers[2], window_size, dropout, step_mode=step_mode) 
        
        self.patch_embed4 = SpikingPatchEmbed3D(embed_dim * 4, embed_dim * 8, patch_size=(2,2,2), step_mode=step_mode)
        self.feature_stage = SpikingSwinTransformerStage3D(
            embed_dim * 8, num_heads[3], layers[3], window_size, dropout, step_mode=step_mode)

        self.patch_expand3 = SpikingPatchExpand3D(embed_dim * 8, embed_dim * 4, dropout=dropout, step_mode=step_mode)
        self.up_stage3 = SpikingSwinTransformerStage3D(
            embed_dim * 4, num_heads[2], layers[2], window_size, dropout, step_mode=step_mode)
        self.converge3 = SpikingAddConverge3D(embed_dim * 4, step_mode=step_mode)
        
        self.patch_expand2 = SpikingPatchExpand3D(embed_dim * 4, embed_dim * 2, dropout=dropout, step_mode=step_mode)
        self.up_stage2 = SpikingSwinTransformerStage3D(
            embed_dim * 2, num_heads[1], layers[1], window_size, dropout, step_mode=step_mode)
        self.converge2 = SpikingAddConverge3D(embed_dim * 2, step_mode=step_mode)

        self.patch_expand1 = SpikingPatchExpand3D(embed_dim * 2, embed_dim, dropout=dropout, step_mode=step_mode)
        self.up_stage1 = SpikingSwinTransformerStage3D(
            embed_dim, num_heads[0], layers[0], window_size, dropout, step_mode=step_mode)
        self.converge1 = SpikingAddConverge3D(embed_dim, step_mode=step_mode)


        self.final_expand = FinalSpikingPatchExpand3D(embed_dim, embed_dim // 3, kernel_size=4, stride=4, dropout=dropout, step_mode=step_mode)

        self.readout = layer.Conv3d(embed_dim // 3, num_classes, kernel_size=1, step_mode=step_mode)
        
        functional.set_step_mode(self, step_mode)

    def forward(self, x_seq):
        functional.reset_net(self)
        
        # step_mode = 's': x shape is [B, C, D, H, W]       
        if self.step_mode == 's':
        
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
                e3 = self.patch_embed3(e2)
                e3 = self.down_stage3(e3)               # e3 shape: [B, 384, 8, 8, 8]

                feature = self.patch_embed3(e3)
                feature = self.feature_stage(feature)       # e3 shape: [B, 768, 4, 4, 4]

                d3 = self.patch_expand3(feature)      # d3 shape: [B, 384, 8, 8, 8]
                d3 = self.up_stage3(d3)              # d3 shape after decode: [B, 384, 8, 8, 8]
                d3 = self.converge3(d3,e3)        # d3 shape after converge: [B, 384, 8, 8, 8]
                
                d2 = self.patch_expand2(d3)      # d3 shape: [B, 192, 16, 16, 16]
                d2 = self.up_stage2(d2)              # d3 shape after decode: [B, 192, 16, 16, 16]
                d2 = self.converge2(d2,e2)        # d3 shape after converge: [B, 192, 16, 16, 16]

                d1 = self.patch_expand1(d2)            # d2 shape: [B, 96, 32, 32, 32] 
                d1 = self.up_stage1(d1)                 # d2 shape after decode: [B, 96, 32, 32, 32]  
                d1 = self.converge1(d1,e1)       # d2 shape after converge: [B, 96, 32, 32, 32]

                d0 = self.final_expand(d1)            # d1 shape: [B, 32, 128, 128, 128]

                out_t = self.readout(d0)          # out_t shape: [B, 3, 128, 128, 128]
                spike_sum = out_t if spike_sum is None else spike_sum + out_t

            return spike_sum / self.T
        
        # step_mode = 'm': x shape is [T, B, C, D, H, W]
        elif self.step_mode == 'm':
            x = x_seq
            
            if x.dim() != 6:
                raise ValueError(f"Input shape must be 6D, but got {x.dim()}D")

            e1 = self.patch_embed1(x)
            e1 = self.down_stage1(e1)

            e2 = self.patch_embed2(e1)
            e2 = self.down_stage2(e2)

            e3 = self.patch_embed3(e2)
            e3 = self.down_stage3(e3)

            feature = self.patch_embed4(e3)
            feature = self.feature_stage(feature)

            d3 = self.patch_expand3(feature)
            d3 = self.up_stage3(d3)
            d3 = self.converge3(d3, e3)
            
            d2 = self.patch_expand2(d3)
            d2 = self.up_stage2(d2)
            d2 = self.converge2(d2, e2)

            d1 = self.patch_expand1(d2)
            d1 = self.up_stage1(d1)
            d1 = self.converge1(d1, e1)

            d0 = self.final_expand(d1)
            out = self.readout(d0)
            out = out.mean(0)
            return out