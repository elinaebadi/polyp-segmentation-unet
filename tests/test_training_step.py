
import torch
import pandas as pd

from torch.utils.data import DataLoader

from config import SPLITS_FILE, BATCH_SIZE, SEED
from src.dataset import PolypDataset
from src.model import UNetSmall
from src.losses import CombinedLoss
from src.metrics import calculate_metrics


# 1. Set random seed
torch.manual_seed(SEED)

# 2. Use CPU for testing
device = torch.device("cpu")

# 3. Load training data
df = pd.read_csv(SPLITS_FILE)
train_df = df[df["split"] == "train"]

train_dataset = PolypDataset(train_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 4. Get one real batch
images, masks = next(iter(train_loader))

images = images.to(device)
masks = masks.to(device)

print("Input shape:", images.shape)
print("Mask shape:", masks.shape)

# 5. Initialize model, loss and optimizer
model = UNetSmall().to(device)
criterion = CombinedLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

model.train()

# 6. Forward pass
logits = model(images)

assert logits.shape == masks.shape

print("Output shape:", logits.shape)

# 7. Calculate loss
loss = criterion(logits, masks)

assert torch.isfinite(loss).item()

print(f"Initial Loss: {loss.item():.4f}")

# 8. Backward pass
optimizer.zero_grad()
loss.backward()

gradients = [
    p.grad for p in model.parameters()
    if p.grad is not None
]

assert len(gradients) > 0
assert all(torch.isfinite(g).all().item() for g in gradients)
assert any(g.abs().sum().item() > 0 for g in gradients)

print("Gradients calculated successfully!")

# 9. Save weights before optimizer step
old_weights = [
    p.detach().clone()
    for p in model.parameters()
]

# 10. Update model weights
optimizer.step()

weights_changed = any(
    not torch.equal(old, new.detach())
    for old, new in zip(old_weights, model.parameters())
)

assert weights_changed

print("Model weights updated:", weights_changed)

# 11. Calculate initial segmentation metrics
metrics = calculate_metrics(
    logits.detach(),
    masks
)

print("\nInitial Metrics:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nTraining step test successful!")
