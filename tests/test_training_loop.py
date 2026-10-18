
import torch
import pandas as pd

from torch.utils.data import DataLoader

from config import SPLITS_FILE, BATCH_SIZE, SEED

from src.dataset import PolypDataset
from src.model import UNetSmall
from src.losses import CombinedLoss
from src.train import train_one_epoch, validate_one_epoch


# Reproducibility
torch.manual_seed(SEED)

# Device
device = torch.device("cpu")

# Load data
df = pd.read_csv(SPLITS_FILE)

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

# Create datasets
train_dataset = PolypDataset(train_df)
val_dataset = PolypDataset(val_df)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Initialize model
model = UNetSmall().to(device)

criterion = CombinedLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Test training on only 2 batches
print("Testing Training Loop...")

train_loss = train_one_epoch(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    max_batches=2
)

print(f"Train Loss: {train_loss:.4f}")

# Test validation on only 2 batches
print("\nTesting Validation Loop...")

val_results = validate_one_epoch(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device,
    max_batches=2
)

for name, value in val_results.items():
    print(f"{name}: {value:.4f}")

# Check results
assert torch.isfinite(torch.tensor(train_loss))
assert torch.isfinite(torch.tensor(list(val_results.values()))).all()

print("\nTraining and Validation tests successful!")
