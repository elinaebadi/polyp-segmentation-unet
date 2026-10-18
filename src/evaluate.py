
import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    ROOT_DIR,
    OUTPUT_DIR,
    METRICS_DIR,
    SPLITS_FILE,
    BATCH_SIZE,
)

from src.dataset import PolypDataset
from src.model import UNetSmall
from src.losses import CombinedLoss
from src.train import validate_one_epoch


def resolve_path(path):
    """Resolve a path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net segmentation checkpoints."
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split (default: val)."
    )

    source = parser.add_mutually_exclusive_group()

    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint for validation evaluation."
    )

    source.add_argument(
        "--selection",
        type=Path,
        help="JSON containing validation-selected models."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for a single validation checkpoint."
    )

    args = parser.parse_args()

    # 1. Resolve models and thresholds
    if args.checkpoint is not None:

        if args.split != "val":
            parser.error(
                "Direct checkpoint evaluation is validation-only. "
                "For test evaluation, use a frozen selection JSON."
            )

        threshold = (
            0.5 if args.threshold is None
            else args.threshold
        )

        models = {
            "custom": {
                "checkpoint": str(args.checkpoint),
                "threshold": threshold,
            }
        }

    else:
        if args.threshold is not None:
            parser.error(
                "--threshold requires --checkpoint."
            )

        selection_path = resolve_path(
            args.selection
            if args.selection is not None
            else METRICS_DIR / "model_selection.json"
        )

        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Model selection not found: {selection_path}"
            )

        with open(
            selection_path, "r", encoding="utf-8"
        ) as f:
            selection = json.load(f)

        if selection.get("selection_dataset") != "validation":
            raise ValueError(
                "Models must be selected using validation data."
            )

        models = selection["models"]

    if not models:
        raise ValueError("No models provided for evaluation.")

    # 2. Load evaluation dataset
    device = torch.device("cpu")

    splits_df = pd.read_csv(SPLITS_FILE)

    eval_df = splits_df[
        splits_df["split"] == args.split
    ].copy()

    if eval_df.empty:
        raise ValueError(
            f"No images found for split: {args.split}"
        )

    dataset = PolypDataset(eval_df)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    criterion = CombinedLoss()

    print("Device:", device)
    print("Split:", args.split)
    print("Images:", len(dataset))

    # 3. Evaluate each model
    results = []

    for model_name, settings in models.items():

        checkpoint_path = resolve_path(
            settings["checkpoint"]
        )

        threshold = float(settings["threshold"])

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"Invalid threshold for {model_name}"
            )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        model = UNetSmall().to(device)

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )

        model.eval()

        metrics = validate_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )

        result = {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "threshold": threshold,
            "loss": metrics["loss"],
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }

        results.append(result)

        print(f"\nModel: {model_name}")
        print(f"Threshold: {threshold:.2f}")
        print(f"Dice: {metrics['dice']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

    # 4. Create a unique evaluation directory
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )

    run_dir = (
        OUTPUT_DIR
        / "runs"
        / "evaluations"
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # 5. Save new results separately
    results_path = (
        run_dir / f"{args.split}_results.csv"
    )

    pd.DataFrame(results).to_csv(
        results_path,
        index=False,
    )

    print("\nEvaluation completed!")
    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()
