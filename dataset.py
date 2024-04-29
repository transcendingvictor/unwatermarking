import os

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):
    def __init__(
        self, watermark_dataset_id, original_dataset_id, split, transform=None
    ):
        """
        Args:
            watermark_dataset_id (string): Hugging Face dataset identifier for the watermarked images.
            original_dataset_id (string): Hugging Face dataset identifier for the original images.
            split (string): Split of the dataset, e.g., 'train', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.watermark_dataset = load_dataset(watermark_dataset_id, split=split)
        self.original_dataset = load_dataset(original_dataset_id, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.watermark_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        watermark_image = self.watermark_dataset[idx]["image"]
        original_image = self.original_dataset[idx]["image"]

        if self.transform:
            watermark_image = self.transform(watermark_image)
            original_image = self.transform(original_image)

        return watermark_image, original_image


# Natural usage in main training script
# transforms = ToTensor()  # This scales the pixel values to the [0, 1] range
# train_dataset = CustomDataset("transcendingvictor/watermark1_flowers_dataset", "transcendingvictor/original_flowers_dataset", "train", transforms)
# val_dataset = CustomDataset("transcendingvictor/watermark1_flowers_dataset", "transcendingvictor/original_flowers_dataset", "validation", transforms)
