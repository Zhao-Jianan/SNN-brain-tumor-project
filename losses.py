import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss as MonaiDiceLoss

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=1.0, smooth=1e-6):
        """
        :param weight: class weights for CrossEntropyLoss
        :param dice_weight: scaling factor for Dice loss component
        :param smooth: smoothing to avoid division by zero
        """
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, target):
        """
        :param pred: model prediction logits of shape [B, C, D, H, W]
        :param target: ground truth labels of shape [B, D, H, W]
        """
        ce = self.ce_loss(pred, target)

        # Convert target to one-hot format: shape [B, C, D, H, W]
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

        # Apply softmax to logits to get probabilities
        pred_soft = F.softmax(pred, dim=1)

        # Compute Dice loss
        dims = (0, 2, 3, 4)  # batch and spatial dims
        intersection = torch.sum(pred_soft * target_onehot, dims)
        cardinality = torch.sum(pred_soft + target_onehot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_per_class[1:].mean()

        return ce + self.dice_weight * dice_loss
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        :param smooth: smoothing to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = {
            1: 2.0,   # Tumor Core (TC)
            2: 1.0,   # Edema
            3: 2.5    # Enhancing Tumor (ET)
        }

    def forward(self, pred, target):
        """
        :param pred: model prediction logits of shape [B, C, D, H, W]
        :param target: ground truth labels of shape [B, D, H, W]
        """
        num_classes = pred.shape[1]
        
        # Convert target to one-hot format: shape [B, C, D, H, W]
        target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

        # Apply softmax to logits to get probabilities
        pred_soft = F.softmax(pred, dim=1)

        # Compute Dice loss
        dims = (0, 2, 3, 4)  # batch and spatial dims
        intersection = torch.sum(pred_soft * target_onehot, dims)
        cardinality = torch.sum(pred_soft + target_onehot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # Skip background class if needed (i.e., dice_per_class[1:])
        loss = 0.0
        total_weight = 0.0
        for cls in range(1, num_classes):
            weight = self.weights.get(cls, 1.0)  # 若未设定权重，默认权重为 1.0
            loss += weight * (1.0 - dice_per_class[cls])
            total_weight += weight

        return loss / total_weight
    
    
    
class BratsDiceLoss(nn.Module):
    def __init__(self, smooth_nr=0.0, smooth_dr=1e-5, squared_pred=True, sigmoid=True, weights=None):
        super().__init__()
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared_pred = squared_pred
        self.sigmoid = sigmoid

        if weights is None:
            self.weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
            self.weights = weights

    def forward(self, pred, target):
        """
        pred: [B, 3, D, H, W] 模型输出 logits 或概率
        target: [B, 3, D, H, W] one-hot 标签 [TC, WT, ET]
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        if self.squared_pred:
            pred = pred ** 2

        dims = (0, 2, 3, 4)
               
        # 如果pred带背景通道，忽略背景通道，取通道1开始
        if pred.shape[1] == 4:
            pred = pred[:, 1:]
        
        intersection = torch.sum(pred * target, dims)
        cardinality = torch.sum(pred + target, dims)

        dice = (2. * intersection + self.smooth_nr) / (cardinality + self.smooth_dr)
        loss_per_channel = 1 - dice  # shape [3]

        weights = self.weights.to(pred.device)

        weighted_loss = (loss_per_channel * weights).sum()

        return weighted_loss