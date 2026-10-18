
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "data" / "raw" / "Kvasir-SEG"

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

SPLITS_FILE = ROOT_DIR / "data" / "splits.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

# Reproducibility
SEED = 42

# Image settings
IMAGE_SIZE = 128

# Training settings
BATCH_SIZE = 4
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Evaluation settings
THRESHOLD = 0.5
