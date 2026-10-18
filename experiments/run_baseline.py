
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
    NUM_EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    THRESHOLD,
)

from src.dataset import PolypDataset
from src.model import UNetSmall
from src.losses import CombinedLoss
from src.train import train_one_epoch, validate_one_epoch


def main():

    # 1. Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")
    print("Using device:", device)

    # 2. Load dataset splits
    df = pd.read_csv(SPLITS_FILE)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    assert len(train_df) == 800
    assert len(val_df) == 100

    # 3. Create datasets
    train_dataset = PolypDataset(train_df)
    val_dataset = PolypDataset(val_df)

    # 4. Create DataLoaders
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

    # 5. Initialize model
    model = UNetSmall().to(device)

    criterion = CombinedLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 6. Training configuration
    best_val_dice = -1.0
    patience_counter = 0
    history = []

    checkpoint_path, history_path = create_run_paths(
    "baseline"
    )

    print("\nStarting Baseline Training...\n")

    # 8. Main training loop
    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_results = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=THRESHOLD
        )

        val_loss = val_results["loss"]
        val_dice = val_results["dice"]

        # Save metrics
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_results["iou"],
            "val_precision": val_results["precision"],
            "val_recall": val_results["recall"]
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        # Save best model
        if val_dice > best_val_dice + 1e-4:

            best_val_dice = val_dice
            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            status = "BEST"

        else:
            patience_counter += 1
            status = ""

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} "
            f"{status}",
            flush=True
        )

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping triggered!")
            break

    print("\nTraining completed!")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()
