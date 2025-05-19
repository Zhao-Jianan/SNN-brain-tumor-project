import torch
import torch.nn as nn
import torch.nn.functional as F

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
        dice_loss = 1. - dice_per_class.mean()

        return ce + self.dice_weight * dice_loss