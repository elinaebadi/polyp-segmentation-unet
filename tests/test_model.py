
import torch

from src.model import UNetSmall
from config import IMAGE_SIZE, BATCH_SIZE

# Create model
model = UNetSmall()
model.eval()

# Create dummy input
x = torch.randn(
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE
)

# Forward pass
with torch.inference_mode():
    output = model(x)

# Count trainable parameters
num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Trainable parameters:", f"{num_parameters:,}")

# Validate output
assert output.shape == (
    BATCH_SIZE,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE
)

assert torch.isfinite(output).all()

print("\nModel test successful!")
