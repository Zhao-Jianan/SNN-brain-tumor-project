import torch
from monai.transforms import NormalizeIntensity
from spikingjelly.clock_driven.encoding import PoissonEncoder, LatencyEncoder, WeightedPhaseEncoder
from monai.inferers import sliding_window_inference
import torch.nn.functional as F

class TemporalSlidingWindowInference:
    def __init__(
        self,
        patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="gaussian",
        encode_method='none',  # 'poisson', 'latency', 'weighted_phase', 'none'
        T=4,
        num_classes=3
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.mode = mode
        self.T = T
        self.encode_method = encode_method
        self.poisson_encoder = PoissonEncoder()
        self.latency_encoder = LatencyEncoder(self.T)
        self.weighted_phase_encoder = WeightedPhaseEncoder(self.T)
        self.num_classes = num_classes

    def _rescale_0_1(self, patch: torch.Tensor) -> torch.Tensor:
        # patch shape: [B, C, D, H, W]
        # 只在 D,H,W 维度做 min-max，保持 B,C 维度分开
        min_val = patch.amin(dim=[1,2,3], keepdim=True)  # 保留 B,C 维度
        max_val = patch.amax(dim=[1,2,3], keepdim=True)
        scale = (max_val - min_val).clamp(min=1e-5)
        return (patch - min_val) / scale

    def encode_spike_input(self, img_rescale: torch.Tensor) -> torch.Tensor:
        """
        对归一化图像进行脉冲编码，支持 poisson / latency / weighted_phase。
        输入:
            img_rescale: torch.Tensor, shape (B, C, D, H, W), 数值应已在 [0, 1] 区间
        输出:
            spike_tensor: torch.Tensor, shape (T, B, C, D, H, W)
        """
        if self.encode_method == 'poisson':
            # [T, B, C, D, H, W]
            spike = torch.stack([self.poisson_encoder(img_rescale) for _ in range(self.T)], dim=0)

        elif self.encode_method == 'latency':
            # Latency 编码只需 encode 一次
            self.latency_encoder.encode(img_rescale)   # 输入 (B, C, D, H, W)
            spike = self.latency_encoder.spike         # (T, B, C, D, H, W)

        elif self.encode_method == 'weighted_phase':
            # Weighted Phase 编码前需调整强度
            x = img_rescale * (1 - 2 ** (-self.T))      # 重要：训练中也做了
            self.weighted_phase_encoder.encode(x)      # 输入 (B, C, D, H, W)
            spike = self.weighted_phase_encoder.spike.float()  # (T, B, C, D, H, W)
        elif self.encode_method == 'none':
            spike = img_rescale.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)

        else:
            raise NotImplementedError(f"Encoding method '{self.encode_method}' is not implemented.")

        return spike

    def _get_weight_window(self, device):
        if self.mode == "gaussian":
            coords = [torch.arange(s, dtype=torch.float32, device=device) for s in self.patch_size]
            zz, yy, xx = torch.meshgrid(*coords, indexing="ij")
            zz = zz - (self.patch_size[0] - 1) / 2
            yy = yy - (self.patch_size[1] - 1) / 2
            xx = xx - (self.patch_size[2] - 1) / 2
            sigmas = [s / 8 for s in self.patch_size]  # 分别计算 σ_D, σ_H, σ_W
            gaussian = torch.exp(
                -(zz ** 2 / (2 * sigmas[0] ** 2) +
                yy ** 2 / (2 * sigmas[1] ** 2) +
                xx ** 2 / (2 * sigmas[2] ** 2))
            )
            return gaussian
        elif self.mode == "constant":
            return torch.ones(self.patch_size, device=device)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __call__(self, inputs: torch.Tensor, predictor: callable) -> torch.Tensor:
        T, B, C, D, H, W = inputs.shape
        device = inputs.device
        pd, ph, pw = self.patch_size
        stride = [int(r * (1 - self.overlap)) for r in self.patch_size]

        # ---------------------------
        # Step 0: Global rescale before sliding window
        # ---------------------------
        def global_rescale_0_1(img: torch.Tensor) -> torch.Tensor:
            # img: [B, C, D, H, W]
            min_val = img.amin(dim=[2, 3, 4], keepdim=True)
            max_val = img.amax(dim=[2, 3, 4], keepdim=True)
            scale = (max_val - min_val).clamp(min=1e-5)
            return (img - min_val) / scale

        # 使用第0帧的图像进行归一化（不影响时间维度）
        inputs_rescaled = inputs.clone()
        if self.encode_method != 'none':
            inputs_rescaled[0] = global_rescale_0_1(inputs[0])

        # ---------------------------
        # Step 1: Padding if needed
        # ---------------------------
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad = [
            pad_w // 2, pad_w - pad_w // 2,  # W
            pad_h // 2, pad_h - pad_h // 2,  # H
            pad_d // 2, pad_d - pad_d // 2   # D
        ]

        inputs_rescaled = inputs_rescaled.view(-1, D, H, W)
        inputs_rescaled = F.pad(inputs_rescaled, pad=pad, mode="constant", value=0.0)
        inputs_rescaled = inputs_rescaled.view(T, B, C, D + pad_d, H + pad_h, W + pad_w)
        D_pad, H_pad, W_pad = inputs_rescaled.shape[-3:]
        padded = any([pad_d, pad_h, pad_w])
        pad_info = (pad_d, pad_h, pad_w, D, H, W)

        # ---------------------------
        # Step 2: Prepare output tensors
        # ---------------------------
        weight_window = self._get_weight_window(device)
        output = torch.zeros((B, self.num_classes, D_pad, H_pad, W_pad), device=device)
        weight_map = torch.zeros((1, 1, D_pad, H_pad, W_pad), device=device)

        # ---------------------------
        # Step 3: Sliding window
        # ---------------------------
        def get_starts(dim, patch_size, stride):
            starts = list(range(0, dim - patch_size + 1, stride))
            if starts[-1] != dim - patch_size:
                starts.append(dim - patch_size)
            return starts

        z_starts = get_starts(D_pad, pd, stride[0])
        y_starts = get_starts(H_pad, ph, stride[1])
        x_starts = get_starts(W_pad, pw, stride[2])

        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = inputs_rescaled[..., z:z+pd, y:y+ph, x:x+pw]  # [T, B, C, pd, ph, pw]
                    for b_start in range(0, B, self.sw_batch_size):
                        b_end = min(b_start + self.sw_batch_size, B)
                        patch_b = patch[:, b_start:b_end]  # [T, b, C, pd, ph, pw]

                        # Step 3.1: Encode directly using already rescaled intensity
                        patch_img = patch_b[0]  # [b, C, pd, ph, pw]
                        # 不再进行 patch 内 rescale
                        # patch_img = self._rescale_0_1(patch_img)
                        patch_encoded = self.encode_spike_input(patch_img)  # [T, b, C, pd, ph, pw]

                        # Step 3.2: Model inference
                        pred = predictor(patch_encoded)  # [b, C_out, pd, ph, pw]

                        output[b_start:b_end, :, z:z+pd, y:y+ph, x:x+pw] += pred * weight_window
                        weight_map[:, :, z:z+pd, y:y+ph, x:x+pw] += weight_window

        # ---------------------------
        # Step 4: Normalize & remove padding
        # ---------------------------
        weight_map = weight_map.clamp(min=1e-5)
        output = output / weight_map

        if padded:
            pad_d, pad_h, pad_w, D, H, W = pad_info
            d_start = pad_d // 2
            h_start = pad_h // 2
            w_start = pad_w // 2
            output = output[:, :, d_start:d_start + D, h_start:h_start + H, w_start:w_start + W]

        return output
