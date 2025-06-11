import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from matplotlib import gridspec

# 加载 NIfTI 图像并返回 numpy 数组
def load_nifti_image(path):
    nii = nib.load(path)
    return nii.get_fdata()

# 创建 RGB mask（1: Blue, 2: Green, 4: Red）
def create_rgba_mask(mask):
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)  # RGBA format
    rgba[mask == 0] = [0, 0, 0, 0]       # fully transparent
    rgba[mask == 1] = [0, 0, 255, 255]     # necrotic and non-enhancing tumor core (NCR/NET — label 1) # Blue, opaque
    rgba[mask == 2] = [0, 255, 0, 255]   # peritumoral edema (ED — label 2)  Green, opaque
    # rgba[mask == 3] = [255, 0, 0, 255]     # Enhancing - Red, opaque
    rgba[mask == 4] = [255, 0, 0, 255]     # GD-enhancing tumor (ET — label 4) # Red, opaque
    return rgba


def select_best_slice(mask):
    best_score = (0, 0)  # (标签种类数量, 标签面积)
    best_slice_idx = 0
    for i in range(mask.shape[2]):
        slice_mask = mask[:, :, i]
        labels = np.unique(slice_mask)
        labels = labels[labels != 0]  # 忽略背景
        label_count = len(labels)
        label_area = np.sum(slice_mask > 0)
        score = (label_count, label_area)
        if score > best_score:
            best_score = score
            best_slice_idx = i
    return best_slice_idx



# 可视化函数

def plot_modalities_with_masks(t1, t1ce, t2, flair, gt_mask, pred_mask, slice_idx=80, save_path='output.png'):
    modalities = [t1, t1ce, t2, flair]
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0.01, hspace=0.01)

    for row in range(3):
        for col, (modality, name) in enumerate(zip(modalities, modality_names)):
            ax = plt.subplot(gs[row, col])
            slice_img = modality[:, :, slice_idx]
            ax.imshow(slice_img, cmap='gray')

            if row == 1:
                mask = gt_mask[:, :, slice_idx]
                ax.imshow(create_rgba_mask(mask))
                if col == 0:
                    ax.set_ylabel('GT', fontsize=12)
            elif row == 2:
                mask = pred_mask[:, :, slice_idx]
                ax.imshow(create_rgba_mask(mask))
                if col == 0:
                    ax.set_ylabel('Pred', fontsize=12)
            elif row == 0 and col == 0:
                ax.set_ylabel('Orig', fontsize=12)

            if row == 0:
                ax.set_title(name, fontsize=12)

            ax.axis('off')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04, wspace=0.01, hspace=0.01)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    # 设置数据目录和文件路径
    data_dir = './data/HGG/Brats18_2013_27_1'  # e.g., BraTS_XXXX/
    case_name = os.path.basename(data_dir)

    t1_path = os.path.join(data_dir, case_name + '_t1.nii')
    t1ce_path = os.path.join(data_dir, case_name + '_t1ce.nii')
    t2_path = os.path.join(data_dir, case_name + '_t2.nii')
    flair_path = os.path.join(data_dir, case_name + '_flair.nii')
    gt_mask_path = os.path.join(data_dir, case_name + '_seg.nii')     # ground truth
    pred_mask_path = os.path.join(data_dir, case_name + '_pred_mask.nii') # model prediction

    # 加载图像数据
    t1 = load_nifti_image(t1_path)
    t1ce = load_nifti_image(t1ce_path)
    t2 = load_nifti_image(t2_path)
    flair = load_nifti_image(flair_path)
    gt_mask = load_nifti_image(gt_mask_path).astype(np.uint8)
    pred_mask = load_nifti_image(pred_mask_path).astype(np.uint8)
    save_dir = './inference/'
    save_path = os.path.join(save_dir, case_name + '_output3.png')

    # 可视化中间层 (中间 slice 通常是肿瘤区域)
    best_slice = select_best_slice(gt_mask)
    plot_modalities_with_masks(t1, t1ce, t2, flair, gt_mask, pred_mask, slice_idx=best_slice, save_path=save_path)
    
if __name__ == '__main__':
    main()
    print("Visualization completed and saved to output.png")
