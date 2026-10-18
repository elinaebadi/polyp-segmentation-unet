
import torch

from src.metrics import calculate_metrics


# Ground truth mask
targets = torch.tensor(
    [[[[1, 1],
       [0, 0]]]],
    dtype=torch.float32
)

# Case 1: Perfect prediction
perfect_logits = torch.tensor(
    [[[[10, 10],
       [-10, -10]]]],
    dtype=torch.float32
)

# Case 2: Partial prediction
partial_logits = torch.tensor(
    [[[[10, -10],
       [-10, -10]]]],
    dtype=torch.float32
)

# Calculate metrics
perfect_metrics = calculate_metrics(
    perfect_logits,
    targets
)

partial_metrics = calculate_metrics(
    partial_logits,
    targets
)

print("Perfect Prediction:")
for name, value in perfect_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nPartial Prediction:")
for name, value in partial_metrics.items():
    print(f"{name}: {value:.4f}")

# Validate results
assert all(
    abs(value - 1.0) < 1e-6
    for value in perfect_metrics.values()
)

assert abs(partial_metrics["dice"] - 2/3) < 1e-6
assert abs(partial_metrics["iou"] - 0.5) < 1e-6
assert abs(partial_metrics["precision"] - 1.0) < 1e-6
assert abs(partial_metrics["recall"] - 0.5) < 1e-6

print("\nMetrics test successful!")
