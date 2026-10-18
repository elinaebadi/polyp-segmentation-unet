
import torch

from src.losses import DiceLoss, CombinedLoss

# Create sample binary masks
masks = torch.randint(
    0, 2, (2, 1, 128, 128)
).float()

# Simulate an untrained model
random_logits = torch.zeros_like(masks)

# Simulate almost perfect predictions
good_logits = torch.where(
    masks == 1,
    torch.full_like(masks, 10.0),
    torch.full_like(masks, -10.0)
)

# Initialize loss functions
dice_fn = DiceLoss()
combined_fn = CombinedLoss()

# Calculate losses
random_dice = dice_fn(random_logits, masks)
random_loss = combined_fn(random_logits, masks)

good_dice = dice_fn(good_logits, masks)
good_loss = combined_fn(good_logits, masks)

print(f"Random Dice Loss: {random_dice.item():.4f}")
print(f"Random Combined Loss: {random_loss.item():.4f}")

print(f"Good Dice Loss: {good_dice.item():.4f}")
print(f"Good Combined Loss: {good_loss.item():.4f}")

# Check loss behavior
assert torch.isfinite(random_loss)
assert torch.isfinite(good_loss)
assert good_loss < random_loss

print("\nLoss functions test successful!")
