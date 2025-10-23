import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossMultiClass(nn.Module):
    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, preds, targets):
        """
        preds: logits [B,C,H,W]
        targets: integer masks [B,H,W]
        """
        preds = F.softmax(preds, dim=1)  # --> Normalize
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # [B,H,W,C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B,C,H,W]

        intersection = (preds * targets_onehot).sum(dim=(2, 3))  # Calculate intersection between prediction and GT.
        union = preds.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))  # Calculate the union of the two.
        dice = (2.0 * intersection + self.eps) / (union + self.eps)  # Value between 1 (100% overlap) and 0 (No overlap)
        return 1 - dice.mean()  # Converts to loss --> 0 Means good overlap


class ComboLoss(nn.Module):
    """CrossEntropy + Dice Loss"""
    # Alpha = 1 --> CrossEntropy
    # Alpha = 0 --> DiceLoss
    # Alpha between 0 and 1 --> Mix of the two

    def __init__(self, num_classes, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()  # Build in Cross Entropy method
        self.dice = DiceLossMultiClass(num_classes)  # Custom dice-loss for multi-class
        self.alpha = alpha

    def forward(self, preds, targets):
        return self.alpha * self.ce(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)
