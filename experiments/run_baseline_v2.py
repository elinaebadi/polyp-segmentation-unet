
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from src.run_utils import create_run_paths

from config import (
    SEED,
    SPLITS_FILE,
    BATCH_SIZE,
    LEARNING_RATE,
    THRESHOLD,
)

from src.dataset import PolypDataset
from src.model import UNetSmall
from src.losses import CombinedLoss
from src.train import train_one_epoch, validate_one_epoch


# V2 Experiment Settings
MAX_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 5
MIN_DELTA = 1e-4


def main():

    # 1. Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")
    print("Using device:", device)

    # 2. Create isolated experiment directory
    checkpoint_path, history_path = create_run_paths(
        "baseline_v2"
    )

    print("Experiment directory:", checkpoint_path.parent)

    # 3. Load dataset
    df = pd.read_csv(SPLITS_FILE)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    assert len(train_df) == 800
    assert len(val_df) == 100

    # 4. Datasets
    train_dataset = PolypDataset(train_df)
    val_dataset = PolypDataset(val_df)

    # 5. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 6. Model
    model = UNetSmall().to(device)

    criterion = CombinedLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 7. Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        threshold=MIN_DELTA,
        threshold_mode="abs",
        min_lr=1e-6
    )

    # 8. Training state
    best_val_dice = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nStarting Baseline V2 Training...\n")

    # 9. Main training loop
    for epoch in range(1, MAX_EPOCHS + 1):

        current_lr = optimizer.param_groups[0]["lr"]

        # Training
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        # Validation
        val_results = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=THRESHOLD
        )

        val_loss = val_results["loss"]
        val_dice = val_results["dice"]

        # Check improvement
        improved = val_dice > best_val_dice + MIN_DELTA

        if improved:

            best_val_dice = val_dice
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            status = "BEST"

        else:
            patience_counter += 1
            status = ""

        # Update learning rate
        scheduler.step(val_dice)

        next_lr = optimizer.param_groups[0]["lr"]

        # Save training history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_results["iou"],
            "val_precision": val_results["precision"],
            "val_recall": val_results["recall"],
            "learning_rate": current_lr
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        # Print progress
        print(
            f"Epoch [{epoch}/{MAX_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.6f} "
            f"{status}",
            flush=True
        )

        # Display LR reduction
        if next_lr < current_lr:
            print(
                f"Learning Rate reduced: "
                f"{current_lr:.6f} -> {next_lr:.6f}"
            )

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping triggered!")
            break

    print("\nTraining completed!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Total Epochs: {len(history)}")
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()
