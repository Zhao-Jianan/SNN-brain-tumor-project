import torch
import nibabel as nib
import numpy as np
import os
from spiking_swin_model import SpikingSwinTransformer3D
import torch.nn.functional as F
from config import T, device
from spikingjelly.activation_based.encoding import PoissonEncoder
from config import device


window_size = (128, 128, 128)
stride = (64, 64, 64)

modalities = ['t1', 't1ce', 't2', 'flair']

class PoissonEncoderWrapper:
    def __init__(self, T):
        self.T = T
        self.encoder = PoissonEncoder()

    def __call__(self, img):
        # img: Tensor [C, D, H, W]
        # 返回Tensor [T, C, D, H, W]
        return torch.stack([self.encoder(img) for _ in range(self.T)], dim=0)  # 一次性生成 [T, C, D, H, W]

def zscore_normalize(img):
    nonzero = img > 0
    if np.any(nonzero):
        mean = img[nonzero].mean()
        std = img[nonzero].std()
        img[nonzero] = (img[nonzero] - mean) / (std + 1e-8)
    else:
        img[:] = 0
    return img

def load_modalities(case_dir, case_name):
    imgs = []
    for mod in modalities:
        path = os.path.join(case_dir, f'{case_name}_{mod}.nii')
        img = nib.load(path).get_fdata()
        img = zscore_normalize(img)
        img = np.transpose(img, (2,0,1))  # 转成 D,H,W顺序
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)  # (4, D, H, W)
    imgs = torch.tensor(imgs).float()
    return imgs



def sliding_window_inference(image, model, encoder, window_size=(128, 128, 128), stride=(64, 64, 64), T=8, num_classes=4):
    """
    Sliding window inference for SNN segmentation
    """
    device = next(model.parameters()).device
    model.eval()

    C, D, H, W = image.shape
    pd, ph, pw = window_size
    sd, sh, sw = stride

    output_probs = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((1, D, H, W), dtype=torch.float32, device=device)

    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                patch = image[:, d:d+pd, h:h+ph, w:w+pw]
                x_seq = encoder(patch).to(device)  # [T, C, pd, ph, pw]

                with torch.no_grad():
                    out = model(x_seq)  # [1, C, pd, ph, pw]
                    out = torch.softmax(out, dim=1)

                output_probs[:, d:d+pd, h:h+ph, w:w+pw] += out.squeeze(0)
                count_map[:, d:d+pd, h:h+ph, w:w+pw] += 1

    count_map[count_map == 0] = 1
    output_probs /= count_map
    return torch.argmax(output_probs, dim=0).cpu()


def main():
    # 路径配置
    case_dir = "./data/HGG/Brats18_2013_2_1"  # 修改为你的case文件夹
    case_name = os.path.basename(case_dir)

    # 模型加载
    model_ckpt = "./checkpoint/best_model_fold1.pth"  # 修改为你的权重路径
    model = SpikingSwinTransformer3D(T=T).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()
    model.num_classes = 4  # 根据你训练类别修改

    # 编码器
    encoder = PoissonEncoderWrapper(T=T)

    # 加载图像数据
    img = load_modalities(case_dir, case_name)  # (4, D, H, W)

    # 滑窗推理
    pred_mask = sliding_window_inference(img, model, encoder, window_size, stride)  # (D,H,W)

    # 转回原始形状（如果pad了，裁剪回去）
    original_img_path = os.path.join(case_dir, f'{case_name}_t1.nii')
    original_img = nib.load(original_img_path)
    orig_shape = original_img.shape
    # 注意原图是 (H,W,D) 这里要对应转换
    # pred_mask 是 (D,H,W)
    d0, h0, w0 = pred_mask.shape
    # 若有pad，裁剪
    if d0 > orig_shape[2] or h0 > orig_shape[0] or w0 > orig_shape[1]:
        pred_mask = pred_mask[:orig_shape[2], :orig_shape[0], :orig_shape[1]]

    # 转置到 (H,W,D)
    pred_mask = np.transpose(pred_mask, (1, 2, 0))

    # 保存预测结果
    pred_nii = nib.Nifti1Image(pred_mask.astype(np.uint8), original_img.affine, original_img.header)
    nib.save(pred_nii, os.path.join(case_dir, f'{case_name}_pred_mask.nii.gz'))
    print(f"预测mask保存至: {os.path.join(case_dir, f'{case_name}_pred_mask.nii.gz')}")

if __name__ == "__main__":
    main()
