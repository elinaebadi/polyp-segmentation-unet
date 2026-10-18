
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Spatial and channel dimensions
        dims = (1, 2, 3)

        # Calculate intersection
        intersection = (probs * targets).sum(dim=dims)

        # Calculate denominator
        denominator = (
            probs.sum(dim=dims) +
            targets.sum(dim=dims)
        )

        # Calculate Dice coefficient
        dice = (
            (2 * intersection + self.smooth) /
            (denominator + self.smooth)
        )

        # Average loss across batch
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined BCE and Dice Loss.
    """

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss
