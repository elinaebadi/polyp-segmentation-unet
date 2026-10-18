
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from config import ROOT_DIR
from config import IMAGE_SIZE


def preprocess_image_mask(image_path, mask_path):
    """
    Load and preprocess an image-mask pair.

    Returns:
        image: float32 array (H, W, 3), range [0, 1]
        mask: float32 array (H, W), values {0, 1}
    """

    # 1. Load image and mask
    with Image.open(image_path) as img, Image.open(mask_path) as msk:

        if img.size != msk.size:
            raise ValueError(
                f"Image/mask size mismatch: {image_path}"
            )

        # 2. Convert color modes
        image = img.convert("RGB")
        mask = msk.convert("L")

        # 3. Resize
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

    # 4. Convert to NumPy arrays
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint8)

    # 5. Normalize image
    image = image / 255.0

    # 6. Binarize mask
    mask = (mask > 127).astype(np.float32)

    return image, mask


class PolypDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        # Get image and mask paths
        row = self.dataframe.iloc[idx]

        image_path = ROOT_DIR / row["image_path"]
        mask_path = ROOT_DIR / row["mask_path"]

        # Apply preprocessing
        image, mask = preprocess_image_mask(
            image_path,
            mask_path
        )

        # Convert image: HWC -> CHW
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.contiguous()

        # Convert mask: HW -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

class AugmentedPolypDataset(PolypDataset):
    """
    Polyp dataset with online data augmentation.
    Augmentation is applied during training only.
    """

    def __getitem__(self, idx):

        # Load preprocessed image and mask
        image, mask = super().__getitem__(idx)

        # 1. Random horizontal flip
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        # 2. Random vertical flip
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        # 3. Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()

        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])

        # 4. Random brightness adjustment
        if torch.rand(1).item() < 0.5:

            brightness_factor = (
                0.9 + torch.rand(1).item() * 0.2
            )

            image = image * brightness_factor

        # 5. Random contrast adjustment
        if torch.rand(1).item() < 0.5:

            contrast_factor = (
                0.9 + torch.rand(1).item() * 0.2
            )

            mean = image.mean(dim=(1, 2), keepdim=True)

            image = (
                image - mean
            ) * contrast_factor + mean

        # Keep image values in [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image.contiguous(), mask.contiguous()
