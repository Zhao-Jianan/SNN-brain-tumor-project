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


def main():
    data_dir = 'Z:/Datasets/archive/MICCAI_BraTS_2018_Data_Validation/Brats18_CBICA_AAM_1'  # e.g., BraTS_XXXX/
    case_name = os.path.basename(data_dir)

    gt_mask_path = os.path.join(data_dir, case_name + '.nii.gz')     # ground truth
    pred_mask_path = os.path.join(data_dir, case_name + '_pred_mask.nii.gz') # model prediction _pred_mask_constant_05.nii
    

    dice_results = compute_dice_from_nifti(pred_mask_path, gt_mask_path)
    print(dice_results)

if __name__ == "__main__":
    main()