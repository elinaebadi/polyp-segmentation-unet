
import torch


@torch.no_grad()
def calculate_metrics(logits, targets, threshold=0.5):
    """
    Calculate binary segmentation metrics.

    Args:
        logits: Model outputs [B, 1, H, W]
        targets: Ground truth masks [B, 1, H, W]
        threshold: Probability threshold

    Returns:
        Dictionary containing mean Dice, IoU,
        Precision, and Recall.
    """

    if logits.shape != targets.shape:
        raise ValueError("Logits and targets must have the same shape.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    # Convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Convert probabilities to binary predictions
    preds = probs > threshold
    targets = targets > 0.5

    # Dimensions to sum over
    dims = (1, 2, 3)

    # True Positives
    tp = (preds & targets).sum(dim=dims).float()

    # False Positives
    fp = (preds & ~targets).sum(dim=dims).float()

    # False Negatives
    fn = (~preds & targets).sum(dim=dims).float()

    # Handle masks where both prediction and target are empty
    both_empty = (tp + fp + fn) == 0

    # Dice Score
    dice_denom = 2 * tp + fp + fn
    dice = torch.where(
        both_empty,
        torch.ones_like(tp),
        (2 * tp) / dice_denom.clamp(min=1)
    )

    # Intersection over Union
    iou_denom = tp + fp + fn
    iou = torch.where(
        both_empty,
        torch.ones_like(tp),
        tp / iou_denom.clamp(min=1)
    )

    # Precision
    precision = torch.where(
        both_empty,
        torch.ones_like(tp),
        tp / (tp + fp).clamp(min=1)
    )

    # Recall
    recall = torch.where(
        both_empty,
        torch.ones_like(tp),
        tp / (tp + fn).clamp(min=1)
    )

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item()
    }
