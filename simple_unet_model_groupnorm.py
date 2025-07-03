import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base
from config import config as cfg

# 使用spikingjelly的多步模式
# class LayerNorm3D(base.MemoryModule):
#     """
#     支持单步（step_mode='s'）和多步（step_mode='m'）模式。
#     """

#     def __init__(self, num_channels, step_mode='s'):
#         super().__init__()
#         self.norm = nn.LayerNorm(num_channels)
#         assert step_mode in ('s', 'm'), "step_mode must be 's' or 'm'"
#         self.step_mode = step_mode

#     def forward(self, x):
#         if self.step_mode == 's':
#             # 单步输入 [B, C, D, H, W]
#             x = x.permute(0, 2, 3, 4, 1).contiguous()  # -> [B, D, H, W, C]
#             x = self.norm(x)
#             x = x.permute(0, 4, 1, 2, 3).contiguous()  # -> [B, C, D, H, W]
#             return x

#         elif self.step_mode == 'm':
#             # 多步输入 [T, B, C, D, H, W]
#             T, B, C, D, H, W = x.shape
#             x = x.view(T * B, C, D, H, W)  # 合并时间和batch
#             x = x.permute(0, 2, 3, 4, 1).contiguous()  # -> [T*B, D, H, W, C]
#             x = self.norm(x)
#             x = x.permute(0, 4, 1, 2, 3).contiguous()  # -> [T*B, C, D, H, W]
#             x = x.view(T, B, C, D, H, W)   # 拆回 [T, B, C, D, H, W]
#             return x



# # 模型结构
# def create_3d_shift_mask(window_size, shift_size):
#     """
#     掩码生成，返回形状[N, N]的bool mask
#     """
#     D, H, W = window_size
#     Sd, Sh, Sw = shift_size
#     if Sd == 0 and Sh == 0 and Sw == 0:
#         return None  # 无需掩码
    
#     img_mask = torch.zeros((1, D, H, W))

#     cnt = 0
#     d_slices = (slice(0, D - Sd), slice(D - Sd, D)) if Sd > 0 else (slice(0, D),)
#     h_slices = (slice(0, H - Sh), slice(H - Sh, H)) if Sh > 0 else (slice(0, H),)
#     w_slices = (slice(0, W - Sw), slice(W - Sw, W)) if Sw > 0 else (slice(0, W),)

#     for d in d_slices:
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, d, h, w] = cnt
#                 cnt += 1

#     mask_windows = img_mask.contiguous().view(-1)
#     attn_mask = mask_windows[:, None] != mask_windows[None, :]  # (N, N)
#     attn_mask = attn_mask.bool().unsqueeze(0).unsqueeze(0)  # -> [1, 1, N, N]
#     attn_mask_float = torch.zeros_like(attn_mask, dtype=torch.float32)
#     attn_mask_float.masked_fill_(attn_mask, float('-inf'))
#     return attn_mask_float


# class SpikingShiftedWindowAttention3D(base.MemoryModule):
#     def __init__(self, embed_dim: int, num_heads: int, window_size=(2, 2, 2), shift_size=None, dropout=0.1, step_mode='s'):
#         super().__init__()
#         self.embed_dim = embed_dim                  # 总的 embedding 维度
#         self.num_heads = num_heads                  # 多头注意力中 head 的个数
#         self.window_size = window_size              # 注意力窗口大小 (D, H, W)
#         self.shift_size = shift_size or tuple(ws // 2 for ws in window_size)
#         self.head_dim = embed_dim // num_heads      # 每个头的维度
#         # assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
#         self.scale = self.head_dim ** -0.5          # 缩放因子，防止 softmax 爆炸
#         self.step_mode = step_mode                  # 's' 单步 / 'm' 多步

#         assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
#         assert len(window_size) == 3, "window_size 必须为3维tuple"
#         assert len(self.shift_size) == 3, "shift_size 必须为3维tuple"

        
#         # 线性变换产生 QKV
#         self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
#         self.proj = nn.Linear(embed_dim, embed_dim)  # 输出投影

#         # dropout 和脉冲神经元（LIF）
#         self.dropout = layer.Dropout(dropout, step_mode=step_mode)
#         # self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
#         self.sn = neuron.ParametricLIFNode(
#             init_tau=2.0,
#             decay_input=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode)

#         # 创建 attention mask（用于 shifted window）
#         self.attn_mask = create_3d_shift_mask(window_size, self.shift_size)

#     def _window_partition(self, x: torch.Tensor):
#         """
#         将输入划分为窗口
#         输入: [B, C, D, H, W]
#         输出: [B*num_windows, window_volume, C]
#         """
#         B, C, D, H, W = x.shape
#         wd, wh, ww = self.window_size
#         x = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
#         x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
#         x = x.view(-1, C, wd * wh * ww).transpose(1, 2).contiguous()  # [B*num_win, N, C]
#         return x, B, D, H, W

#     def _window_reverse(self, x_windows: torch.Tensor, B, D, H, W):
#         """
#         将窗口恢复为原始图像形状
#         """
#         wd, wh, ww = self.window_size
#         C = x_windows.shape[2]
#         x_windows = x_windows.transpose(1, 2).contiguous().view(-1, C, wd, wh, ww)
#         x_windows = x_windows.view(B, D // wd, H // wh, W // ww, C, wd, wh, ww)
#         x_windows = x_windows.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
#         x = x_windows.view(B, C, D, H, W)
#         return x


#     def _attention_forward(self, x_windows: torch.Tensor, batch_size: int):
#         """
#         统一的注意力计算逻辑
#         x_windows: [batch_size * num_windows, N, C]
#         batch_size: 单步时是 B，多步时是 T*B
#         返回: [batch_size * num_windows, N, C] 处理后窗口张量
#         """       
#         # QKV分解
#         qkv = self.qkv(x_windows)
#         qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*num_windows, H, N, N]

#         # # 使用相对位置索引取出偏置值：shape [N*N, H]
#         # relative_position_bias = self.relative_position_table[self.relative_position_index.view(-1)]  # (N*N, H)

#         # # 重新 reshape 为 [H, N, N]，即每个 head 的偏置矩阵
#         # relative_position_bias = rearrange(relative_position_bias, '(n1 n2) h -> h n1 n2', 
#         #                                    n1=self.relative_position_index.shape[0])

#         # # 添加到注意力矩阵中，自动广播为 [B, H, N, N]
#         # attn = attn + relative_position_bias.unsqueeze(0)  # [1, H, N, N]
        
#         if self.attn_mask is not None:
#             attn = attn + self.attn_mask.to(attn.device)
#         attn = torch.softmax(attn, dim=-1)        
#         out = torch.matmul(attn, v)  # [batch_size*num_windows, num_heads, N, head_dim]        
#         out = out.permute(0, 2, 1, 3).contiguous().view(out.size(0), out.size(2), self.embed_dim)

#         out = self.proj(out)       
#         return out  # [batch_size*num_windows, N, C]
  

#     def forward_single_step(self, x: torch.Tensor):
#         """
#         单步输入：[B, C, D, H, W]
#         """
#         B, C, D, H, W = x.shape
#         sd, sh, sw = self.shift_size

#         if any(s > 0 for s in self.shift_size):
#             x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))

#         x_windows, B, D, H, W = self._window_partition(x)

#         out = self._attention_forward(x_windows, B)
#         out = self.sn(out)
#         x = self._window_reverse(out, B, D, H, W)

#         if any(s > 0 for s in self.shift_size):
#             x = torch.roll(x, shifts=(sd, sh, sw), dims=(2, 3, 4))

#         return x

#     def forward_multi_step(self, x: torch.Tensor):
#         """
#         多步输入：[T, B, C, D, H, W]
#         """
#         T, B, C, D, H, W = x.shape
#         wd, wh, ww = self.window_size
#         sd, sh, sw = self.shift_size

#         if any(s > 0 for s in self.shift_size):
#             x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(3, 4, 5))  # shift

#         x = x.view(T * B, C, D, H, W)
#         x_windows, new_B, D, H, W = self._window_partition(x)

#         out = self._attention_forward(x_windows, T * B)

#         # 脉冲激活
#         # out = self.dropout(out)
#         # out = out + x_windows
#         out = out.view(T, B, -1, self.embed_dim)
#         out = self.sn(out)
#         out = out.view(T * B, -1, self.embed_dim)

#         x = self._window_reverse(out, new_B, D, H, W)
#         x = x.view(T, B, C, D, H, W)

#         if any(s > 0 for s in self.shift_size):
#             x = torch.roll(x, shifts=(sd, sh, sw), dims=(3, 4, 5))  # shift back
#         return x

#     def forward(self, x):
#         if self.step_mode == 's':
#             return self.forward_single_step(x)
#         elif self.step_mode == 'm':
#             return self.forward_multi_step(x)
#         else:
#             raise NotImplementedError(f"Unsupported step_mode: {self.step_mode}")


# class SpikingSwinTransformerBlock3D(base.MemoryModule):
#     def __init__(self, embed_dim, mlp_dim, num_heads, window_size=(2, 2, 2), shift=False, dropout=0.1, num_groups=8, step_mode='s'):
#         super().__init__()       
#         shift_size = tuple(ws // 2 for ws in window_size) if shift else (0, 0, 0)

#         # self.norm1 = LayerNorm3D(embed_dim, step_mode=step_mode)
#         self.norm1 = layer.GroupNorm(num_groups=num_groups, num_channels=embed_dim, step_mode=step_mode) # 使用 GroupNorm 替代 LayerNorm
#         self.attn = SpikingShiftedWindowAttention3D(embed_dim, num_heads, window_size, shift_size, dropout, step_mode=step_mode)

#         # self.norm2 = LayerNorm3D(embed_dim, step_mode=step_mode)
#         self.norm2 = layer.GroupNorm(num_groups=num_groups, num_channels=embed_dim, step_mode=step_mode)
#         self.linear1 = layer.Linear(embed_dim, mlp_dim, step_mode=step_mode)
#         # self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
#         self.sn1 = neuron.ParametricLIFNode(
#             init_tau=2.0,
#             decay_input=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode)
#         self.linear2 = layer.Linear(mlp_dim, embed_dim, step_mode=step_mode)
#         # self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)  
#         self.sn2 = neuron.ParametricLIFNode(
#             init_tau=2.0,
#             decay_input=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode)     
#         self.dropout = layer.Dropout(dropout, step_mode=step_mode)

#     def _linear_forward(self, x):
#         x1 = self.linear1(x)
#         x1 = self.sn1(x1)
#         # x1 = self.gelu(x1)
#         x2 = self.linear2(x1)
#         #x2 = self.dropout(x2)
#         output = self.sn2(x2)  
#         output = self.dropout(output)
#         return output

#     def forward(self, x):
#         residual = x
#         x = self.norm1(x)
#         x = self.attn(x)
#         x = x + residual # 残差连接

#         residual = x
#         x = self.norm2(x)
                
#         if self.step_mode == 's':
#             # 单步输入: [B, C, D, H, W]
#             B, C, D, H, W = x.shape
#             x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
#             x = x.view(B, D * H * W, C)  # (B, N, C)
#             x = self._linear_forward(x)
#             x = x.view(B, D, H, W, C)
#             x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
#         elif self.step_mode == 'm':
#             # 多步输入: [T, B, C, D, H, W]
#             T, B, C, D, H, W = x.shape
#             x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # (T, B, D, H, W, C)
#             x = x.view(T * B, D * H * W, C)  # (T*B, N, C)
#             x = self._linear_forward(x)
#             x = x.view(T, B, D, H, W, C)
#             x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # (T, B, C, D, H, W)
#         else:
#             raise NotImplementedError(f"Unsupported step_mode: {self.step_mode}")
        
#         x = x + residual
#         return x

# class SpikingSwinTransformerStage3D(base.MemoryModule):
#     def __init__(self, embed_dim, num_heads, layers, window_size=(2, 2, 2), dropout=0.1, num_groups=8, step_mode='s'):
#         super().__init__()
#         block_list = []
#         for i in range(layers):
#             shift = (i % 2 == 1)  # 奇数 block shift=True，偶数 False
#             block_list.append(
#                 SpikingSwinTransformerBlock3D(
#                     embed_dim=embed_dim,
#                     mlp_dim=embed_dim*4,
#                     num_heads=num_heads,
#                     window_size=window_size,
#                     shift=shift,
#                     dropout=dropout,
#                     num_groups=num_groups,
#                     step_mode=step_mode
#                 )
#             )
#         self.blocks = nn.Sequential(*block_list)

#     def forward(self, x):
#         return self.blocks(x)


class PatchEmbed3D(base.MemoryModule):
    def __init__(self, in_channels, embed_dim, patch_size=(2, 2, 2), num_groups=8):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class SpikingPatchExpand3D(base.MemoryModule):
    def __init__(self, in_channels, out_channels, kernel_size=(2,2,2), stride=2, dropout=0.1, num_groups=8, step_mode='s'):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        # self.norm = LayerNorm3D(out_channels, step_mode=step_mode)
        self.norm = layer.GroupNorm(num_groups=num_groups, num_channels=out_channels, step_mode=step_mode)
        # self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
        self.sn = neuron.ParametricLIFNode(
            init_tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), 
            step_mode=step_mode)
        functional.set_step_mode(self, step_mode)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.sn(x)
        return x


class FinalSpikingPatchExpand3D(base.MemoryModule):
    def __init__(self, in_channels, out_channels, kernel_size=(2,2,2), stride=2, dropout=0.1, num_groups=8, step_mode='s'):
        super().__init__()
        self.conv_transpose = layer.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, step_mode=step_mode)
        # self.norm = LayerNorm3D(out_channels, step_mode=step_mode)
        self.norm = layer.GroupNorm(num_groups=num_groups, num_channels=out_channels, step_mode=step_mode)
        # self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
        # self.sn = neuron.ParametricLIFNode(
        #     init_tau=2.0,
        #     decay_input=True,
        #     v_threshold=1.0,
        #     v_reset=0.0,
        #     surrogate_function=surrogate.ATan(), 
        #     step_mode=step_mode)
        # functional.set_step_mode(self, step_mode)
        self.relu = layer.ReLU(inplace=True)  # 使用 ReLU 激活函数

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.sn(x)
        return x


# class SpikingConcatReduce3D(base.MemoryModule):
#     def __init__(self, in_channels, out_channels, step_mode='s'):
#         super().__init__()
#         concat_channels = in_channels * 2

#         self.norm1 = LayerNorm3D(concat_channels, step_mode=step_mode)
#         self.conv = layer.Conv3d(concat_channels, out_channels, kernel_size=1, step_mode=step_mode)
#         self.norm2 = LayerNorm3D(out_channels, step_mode=step_mode)

#         functional.set_step_mode(self, step_mode)

#     def forward(self, x1, x2):
#         x = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension
#         x = self.norm1(x)
#         x = self.conv(x)
#         x = self.norm2(x)
#         return x

    

class SpikingAddConverge3D(base.MemoryModule):
    def __init__(self, channels, num_groups=8, step_mode='s'):
        super().__init__()
        # self.norm = LayerNorm3D(channels, step_mode=step_mode)
        self.norm = layer.GroupNorm(num_groups=num_groups, num_channels=channels, step_mode=step_mode)
        functional.set_step_mode(self, step_mode)

    def forward(self, x1, x2):
        x = x1 + x2  # skip connection by addition
        x = self.norm(x)
        return x



class SimpleUNet3D(base.MemoryModule):
    def __init__(self, in_channels=4, num_classes=3, embed_dim=cfg.embed_dim, layers=[2, 2, 4, 2], num_heads=cfg.num_heads,
                 window_size=(4,4,4), dropout=0.1, T=8, num_norm_groups=cfg.num_norm_groups, step_mode='s'):
        super().__init__()
        self.T = T
        self.step_mode = step_mode

        self.patch_embed1 = SpikingPatchEmbed3D(
            in_channels, embed_dim, patch_size=(4,4,4), num_groups=num_norm_groups[0], step_mode=step_mode)
        # self.down_stage1 = SpikingSwinTransformerStage3D(
        #     embed_dim, num_heads[0], layers[0], window_size, dropout, num_groups=num_norm_groups[0], step_mode=step_mode)

        self.patch_embed2 = SpikingPatchEmbed3D(
            embed_dim, embed_dim * 2, patch_size=(2,2,2), num_groups=num_norm_groups[1], step_mode=step_mode)
        # self.down_stage2 = SpikingSwinTransformerStage3D(
        #     embed_dim * 2, num_heads[1], layers[1], window_size, dropout, num_groups=num_norm_groups[1], step_mode=step_mode)

        self.patch_embed3 = SpikingPatchEmbed3D(
            embed_dim * 2, embed_dim * 4, patch_size=(2,2,2), num_groups=num_norm_groups[2], step_mode=step_mode)
        # self.down_stage3 = SpikingSwinTransformerStage3D(
        #     embed_dim * 4, num_heads[2], layers[2], window_size, dropout, num_groups=num_norm_groups[2], step_mode=step_mode) 
        
        self.patch_embed4 = SpikingPatchEmbed3D(
            embed_dim * 4, embed_dim * 8, patch_size=(2,2,2), num_groups=num_norm_groups[3], step_mode=step_mode)
        # self.feature_stage = SpikingSwinTransformerStage3D(
        #     embed_dim * 8, num_heads[3], layers[3], window_size, dropout, num_groups=num_norm_groups[3], step_mode=step_mode)

        self.patch_expand3 = SpikingPatchExpand3D(
            embed_dim * 8, embed_dim * 4, dropout=dropout, num_groups=num_norm_groups[2], step_mode=step_mode)
        # self.up_stage3 = SpikingSwinTransformerStage3D(
        #     embed_dim * 4, num_heads[2], layers[2], window_size, dropout, num_groups=num_norm_groups[2], step_mode=step_mode)
        self.converge3 = SpikingAddConverge3D(embed_dim * 4, num_groups=num_norm_groups[2], step_mode=step_mode)
        
        self.patch_expand2 = SpikingPatchExpand3D(
            embed_dim * 4, embed_dim * 2, dropout=dropout, num_groups=num_norm_groups[1], step_mode=step_mode)
        # self.up_stage2 = SpikingSwinTransformerStage3D(
        #     embed_dim * 2, num_heads[1], layers[1], window_size, dropout, num_groups=num_norm_groups[1], step_mode=step_mode)
        self.converge2 = SpikingAddConverge3D(embed_dim * 2, num_groups=num_norm_groups[1], step_mode=step_mode)

        self.patch_expand1 = SpikingPatchExpand3D(
            embed_dim * 2, embed_dim, dropout=dropout, num_groups=num_norm_groups[0], step_mode=step_mode)
        # self.up_stage1 = SpikingSwinTransformerStage3D(
        #     embed_dim, num_heads[0], layers[0], window_size, dropout, num_groups=num_norm_groups[0], step_mode=step_mode)
        self.converge1 = SpikingAddConverge3D(embed_dim, num_groups=num_norm_groups[0], step_mode=step_mode)


        self.final_expand = FinalSpikingPatchExpand3D(
            embed_dim, embed_dim // 3, kernel_size=4, stride=4, dropout=dropout, num_groups=4, step_mode=step_mode)

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
                #e1 = self.down_stage1(e1)               # e1 shape: [B, 96, 32, 32, 32]
                e2 = self.patch_embed2(e1)
                #e2 = self.down_stage2(e2)               # e2 shape: [B, 192, 16, 16, 16]
                e3 = self.patch_embed3(e2)
                #e3 = self.down_stage3(e3)               # e3 shape: [B, 384, 8, 8, 8]

                feature = self.patch_embed4(e3)
                #feature = self.feature_stage(feature)       # e3 shape: [B, 768, 4, 4, 4]

                d3 = self.patch_expand3(feature)      # d3 shape: [B, 384, 8, 8, 8]
                #d3 = self.up_stage3(d3)              # d3 shape after decode: [B, 384, 8, 8, 8]
                d3 = self.converge3(d3,e3)        # d3 shape after converge: [B, 384, 8, 8, 8]
                
                d2 = self.patch_expand2(d3)      # d3 shape: [B, 192, 16, 16, 16]
                #d2 = self.up_stage2(d2)              # d3 shape after decode: [B, 192, 16, 16, 16]
                d2 = self.converge2(d2,e2)        # d3 shape after converge: [B, 192, 16, 16, 16]

                d1 = self.patch_expand1(d2)            # d2 shape: [B, 96, 32, 32, 32] 
                #d1 = self.up_stage1(d1)                 # d2 shape after decode: [B, 96, 32, 32, 32]  
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
            #e1 = self.down_stage1(e1)
            e2 = self.patch_embed2(e1)
            #e2 = self.down_stage2(e2)

            e3 = self.patch_embed3(e2)
            #e3 = self.down_stage3(e3)

            feature = self.patch_embed4(e3)
            #feature = self.feature_stage(feature)

            d3 = self.patch_expand3(feature)
            #d3 = self.up_stage3(d3)
            d3 = self.converge3(d3, e3)
            
            d2 = self.patch_expand2(d3)
            #d2 = self.up_stage2(d2)
            d2 = self.converge2(d2, e2)

            d1 = self.patch_expand1(d2)
            #d1 = self.up_stage1(d1)
            d1 = self.converge1(d1, e1)

            d0 = self.final_expand(d1)
            out = self.readout(d0)
            out = out.mean(0)
            return out
        
        
def main():
    # 测试模型
    model = SpikingSwinUNet3D(in_channels=4, num_classes=3, embed_dim=96, layers=[2, 2, 4, 2], num_heads=[3, 6, 12, 24], window_size=(4, 4, 4), dropout=0.1, T=2, step_mode='s')
    x = torch.randn(2, 1, 4, 128, 128, 128)  # 假设输入是一个 batch 的数据
    output = model(x)
    print(output.shape)  # 输出形状应该是 [1, 3, 128, 128, 128]
    
if __name__ == "__main__":
    main()