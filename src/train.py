
import torch

from src.metrics import calculate_metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    max_batches=None
):
    """
    Train the model for one epoch.

    Returns:
        Average training loss.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, masks) in enumerate(dataloader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move tensors to device
        images = images.to(device)
        masks = masks.to(device)

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, masks)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss!")

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty!")

    return total_loss / total_samples


@torch.inference_mode()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    threshold=0.5,
    max_batches=None
):
    """
    Evaluate model on validation data.

    Returns:
        Dictionary of validation loss and metrics.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall"
    ]

    metric_totals = {
        name: 0.0 for name in metric_names
    }

    for batch_idx, (images, masks) in enumerate(dataloader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        logits = model(images)

        # Calculate validation loss
        loss = criterion(logits, masks)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite validation loss!")

        # Calculate metrics
        metrics = calculate_metrics(
            logits,
            masks,
            threshold=threshold
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        for name in metric_names:
            metric_totals[name] += metrics[name] * batch_size

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation DataLoader is empty!")

    results = {
        "loss": total_loss / total_samples
    }

    for name in metric_names:
        results[name] = metric_totals[name] / total_samples

    return results
