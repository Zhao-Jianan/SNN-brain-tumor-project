import nibabel as nib
import numpy as np
import torch
import os

def load_nifti_as_tensor(path):
    """Load NIfTI and return tensor of shape (D, H, W)"""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.uint8)
    if data.ndim == 4:
        data = data.squeeze()  # in case it's (1, D, H, W)
    return torch.from_numpy(data)

def dice_score(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum().float()
    denominator = pred_mask.sum().float() + gt_mask.sum().float()
    if denominator == 0:
        return torch.tensor(1.0)
    return 2.0 * intersection / denominator

def compute_dice_from_nifti(pred_path, gt_path):
    pred = load_nifti_as_tensor(pred_path)
    gt = load_nifti_as_tensor(gt_path)
    
    # Convert to label space if needed
    pred = pred.cpu()
    gt = gt.cpu()

    # TC: label 1 or 4
    pred_tc = (pred == 1) | (pred == 4)
    gt_tc   = (gt == 1)   | (gt == 4)

    # WT: label 1 or 2 or 4
    pred_wt = (pred == 1) | (pred == 2) | (pred == 4)
    gt_wt   = (gt == 1)   | (gt == 2)   | (gt == 4)

    # ET: label 4
    pred_et = (pred == 4)
    gt_et   = (gt == 4)
    

    print("Sum TC:", pred_tc.sum().item(), "GT TC:", gt_tc.sum().item())
    print("Sum WT:", pred_wt.sum().item(), "GT WT:", gt_wt.sum().item())
    print("Sum ET:", pred_et.sum().item(), "GT ET:", gt_et.sum().item())
    print('Sum NCR/NET:', (pred == 1).sum().item(), "GT NCR/NET:", (gt == 1).sum().item())
    print('Sum ED:', (pred == 2).sum().item(), "GT ED:", (gt == 2).sum().item())
    print('Sum BG:', (pred == 0).sum().item(), "GT BG:", (gt == 0).sum().item())

    dice_tc = dice_score(pred_tc, gt_tc)
    dice_wt = dice_score(pred_wt, gt_wt)
    dice_et = dice_score(pred_et, gt_et)
    mean_dice = (dice_tc + dice_wt + dice_et) / 3

    return {
        "Dice_TC": round(dice_tc.item(), 4),
        "Dice_WT": round(dice_wt.item(), 4),
        "Dice_ET": round(dice_et.item(), 4),
        "Mean_Dice": round(mean_dice.item(), 4),
    }



def batch_compute_dice(gt_dir, pred_dir):
    all_dice_scores = []

    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('_pred_mask.nii.gz'):
            continue

        # 获取病例名
        case_name = pred_file.replace('_pred_mask.nii.gz', '')

        # 构造文件路径
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, case_name + '.nii.gz')

        if not os.path.exists(gt_path):
            print(f"[Warning] Ground truth not found for {case_name}, skipping.")
            continue

        # 计算 Dice
        dice = compute_dice_from_nifti(pred_path, gt_path)
        all_dice_scores.append(dice)

        print(f"{case_name}: {dice}")

    # 计算平均值
    dice_tc = [d["Dice_TC"] for d in all_dice_scores]
    dice_wt = [d["Dice_WT"] for d in all_dice_scores]
    dice_et = [d["Dice_ET"] for d in all_dice_scores]
    mean_dice = [d["Mean_Dice"] for d in all_dice_scores]

    print("\n=== Average Dice Scores ===")
    print(f"Dice_TC Mean: {np.mean(dice_tc):.4f}")
    print(f"Dice_WT Mean: {np.mean(dice_wt):.4f}")
    print(f"Dice_ET Mean: {np.mean(dice_et):.4f}")
    print(f"Mean_Dice:   {np.mean(mean_dice):.4f}")






def main():
    batch_compute = False
    
    if batch_compute:
        # 批量计算 Dice
        gt_dir = './val_pred/nnUNetTrainer'
        pred_dir = './val_pred/test_pred'
        batch_compute_dice(gt_dir, pred_dir)
    else:
        # compute single case Dice
        data_dir = './data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_27_1'
        pred_dir = './pred'
        case_name = os.path.basename(data_dir)

        gt_mask_path = os.path.join(data_dir, case_name + '_seg.nii')     # ground truth
        pred_mask_path = os.path.join(pred_dir, case_name + '_pred_mask.nii.gz') # model prediction _pred_mask_constant_05.nii
        

        dice_results = compute_dice_from_nifti(pred_mask_path, gt_mask_path)
        print(dice_results)

if __name__ == "__main__":
    main()