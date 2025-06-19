import torch
import torch.nn as nn
from typing import Tuple
from spikingjelly.activation_based import neuron, layer, base, surrogate


def create_3d_shift_mask(window_size, shift_size):
    """
    创建 3D shift attention mask，用于模拟 Swin Transformer 中的遮挡机制
    """
    D, H, W = window_size
    Sd, Sh, Sw = shift_size
    cnt = 0
    img_mask = torch.zeros((1, D, H, W))
    
    # 将每个局部区域赋予不同的mask ID，便于后续构建attention遮挡关系
    for d in (slice(0, -Sd), slice(-Sd, None)):
        for h in (slice(0, -Sh), slice(-Sh, None)):
            for w in (slice(0, -Sw), slice(-Sw, None)):
                img_mask[:, d, h, w] = cnt
                cnt += 1
    
    # 拉平成一维，再比较不同token之间的mask id是否一致
    mask_windows = img_mask.reshape(1, -1)
    attn_mask = (mask_windows != mask_windows.transpose(1, 0)).float()
    attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf')).masked_fill(attn_mask == 0, 0.0)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
    return attn_mask


class SpikingShiftedWindowAttention3D(base.MemoryModule):
    def __init__(self, embed_dim: int, num_heads: int, window_size=(2, 2, 2), shift_size=None, dropout=0.1, step_mode='s'):
        super().__init__()
        self.embed_dim = embed_dim                  # 总的 embedding 维度
        self.num_heads = num_heads                  # 多头注意力中 head 的个数
        self.window_size = window_size              # 注意力窗口大小 (D, H, W)
        self.shift_size = shift_size or tuple(ws // 2 for ws in window_size)
        self.head_dim = embed_dim // num_heads      # 每个头的维度
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.scale = self.head_dim ** -0.5          # 缩放因子，防止 softmax 爆炸
        self.step_mode = step_mode                  # 's' 单步 / 'm' 多步

        # 线性变换产生 QKV
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)  # 输出投影

        # dropout 和脉冲神经元（LIF）
        self.dropout = layer.Dropout(dropout, step_mode=step_mode)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)

        # 创建 attention mask（用于 shifted window）
        self.attn_mask = create_3d_shift_mask(window_size, self.shift_size)

    def _window_partition(self, x: torch.Tensor):
        """
        将输入划分为窗口
        输入: [B, C, D, H, W]
        输出: [B*num_windows, window_volume, C]
        """
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        x = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(-1, C, wd * wh * ww).transpose(1, 2)  # [B*num_win, N, C]
        return x, B, D, H, W

    def _window_reverse(self, x_windows: torch.Tensor, B, D, H, W):
        """
        将窗口恢复为原始图像形状
        """
        wd, wh, ww = self.window_size
        C = x_windows.shape[2]
        x_windows = x_windows.transpose(1, 2).contiguous().view(-1, C, wd, wh, ww)
        x_windows = x_windows.view(B, D // wd, H // wh, W // ww, C, wd, wh, ww)
        x_windows = x_windows.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x_windows.view(B, C, D, H, W)
        return x

    def scaled_dot_attn(self, q, k, v, mask=None):
        """
        标准的多头 dot-product attention
        输入: q, k, v [B*win, h, N, d]
        输出: attention 输出 [B*win, h, N, d]
        """
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*win, h, N, N]
        if mask is not None:
            attn = attn + mask.to(attn.device)  # 应用shift attention遮罩
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B*win, h, N, d]
        return out

    def forward_single_step(self, x: torch.Tensor):
        """
        单步输入：[B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))  # shifted window

        x_windows, B, D, H, W = self._window_partition(x)  # [B*num_win, N, C]

        # 生成 Q, K, V 并重构为多头形式
        qkv = self.qkv(x_windows)  # [B*num_win, N, 3*C]
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*num_win, h, N, d]

        # 注意力 + 输出
        out = self.scaled_dot_attn(q, k, v, self.attn_mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.size(0), out.size(2), self.embed_dim)
        out = self.proj(out)
        out = self.dropout(out)
        out = out + x_windows
        out = self.sn(out)  # 脉冲激活

        x = self._window_reverse(out, B, D, H, W)
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(2, 3, 4))  # shift back
        return x

    def forward_multi_step(self, x: torch.Tensor):
        """
        多步输入：[T, B, C, D, H, W]
        """
        T, B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(3, 4, 5))  # shift

        x = x.view(T * B, C, D, H, W)
        x_windows, new_B, D, H, W = self._window_partition(x)

        # QKV分解
        qkv = self.qkv(x_windows)
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = self.scaled_dot_attn(q, k, v, self.attn_mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.size(0), out.size(2), self.embed_dim)
        out = self.proj(out)

        # reshape 回 T, B 格式 + 脉冲激活
        out = out.view(T, B, -1, self.embed_dim)
        out = self.dropout(out)
        out = self.sn(out)
        out = out.view(T * B, -1, self.embed_dim)

        x = self._window_reverse(out, new_B, D, H, W)
        x = x.view(T, B, C, D, H, W)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(3, 4, 5))  # shift back
        return x

    def forward(self, x):
        if self.step_mode == 's':
            return self.forward_single_step(x)
        elif self.step_mode == 'm':
            return self.forward_multi_step(x)
        else:
            raise NotImplementedError(f"Unsupported step_mode: {self.step_mode}")
