"""
This module defines a custom dataset class for loading and processing images from a directory.
It supports filtering specific classes, applying transformations, and returning images with
multi-hot encoded labels.

Classes:
    CustomImageDataset: A PyTorch Dataset for loading images with optional class filtering
                        and transformations.
"""

import json
import os
import random
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.config_utils import BROKEN_FILES_PATH
from vision_inference.logger import Logger


class MGRSImageDataset(Dataset):
    """A custom dataset for loading images with MGRS grid-based multi-hot encoding."""

    def __init__(
        self,
        root_dir: str,
        broken_files: Optional[List[str]] = None,
        root_dir_non_salient: Optional[str] = None,
        salient_regions: List[str] = None,
        transform: Optional[object] = None,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """
        Args:
            root_dir (str): Path to the dataset directory.
            root_dir_non_salient (Optional[str]): Path to the non-salient dataset directory.
            salient_regions (List[str]): List of MGRS regions to consider.
            transform (Optional[object]): Optional transforms to apply to images.
            split (str): One of 'train', 'val', or 'test'.
            train_ratio (float): Ratio of data to use for training.
            val_ratio (float): Ratio of data to use for validation.
            seed (int): Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.salient_regions = sorted(salient_regions or [])

        # Create mapping from region to index
        self.salient_region_indices = {region: i for i, region in enumerate(self.salient_regions)}

        # Set sigmoid parameters
        self.sigmoid_params = self._calculate_sigmoid_params(0.2, 0.05, 0.3, 0.95)

        Logger.log("INFO", f"Sigmoid parameters: k={self.sigmoid_params['k']:.4f}, x0={self.sigmoid_params['x0']:.4f}")

        # Collect images and their corresponding lat/lon files
        self.files = []
        salient_regions_set = set(self.salient_regions)
        broken_file_set = set(broken_files or [])
        for f in os.listdir(root_dir):
            # Iterate through region folders
            print(f"Processing region folder: {f}")
            if os.path.isdir(os.path.join(root_dir, f)) and f in salient_regions_set:
                print(f'Directory Path: {os.path.join(root_dir, f)}')
                region_dir = os.path.join(root_dir, f)
                for file in os.listdir(region_dir):
                    # print(f"Processing file: {file}")
                    if (file.endswith(".png") or file.endswith(".jpg")) and file not in broken_file_set:
                        # The files all start with 'l8_' followed by region and image ID
                        if not file[3].isdigit():
                            continue
                        img_path = os.path.join(region_dir, file)
                        json_path = os.path.join(
                            region_dir, file.rsplit(".", 1)[0] + "_mgrs_counts.json"
                        )
                        if os.path.exists(json_path):
                            self.files.append((img_path, json_path))
                        else:
                            warnings.warn(
                                f"JSON file not found for {img_path}. Skipping this image."
                            )

        if root_dir_non_salient:
            for file in os.listdir(root_dir_non_salient):
                if file.endswith(".png") or file.endswith(".jpg"):
                    img_path = os.path.join(root_dir_non_salient, file)
                    self.files.append((img_path, None))
        # Split dataset
        random.seed(seed)

        # Shuffle files
        random.shuffle(self.files)

        # Calculate split sizes
        total_size = len(self.files)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)

        # Split the data
        if split == "train":
            self.files = self.files[:train_size]
        elif split == "val":
            self.files = self.files[train_size : train_size + val_size]
        else:  # test
            self.files = self.files[train_size + val_size :]

        Logger.log("INFO", f"Total {split} images: {len(self.files)}")

    def _append_broken_file(self, img_path: str) -> None:
        """Append a broken image filename to broken_files.yaml if not already present."""
        file_name = os.path.basename(img_path)
        try:
            existing = set()
            if os.path.exists(BROKEN_FILES_PATH):
                with open(BROKEN_FILES_PATH, "r", encoding="utf-8") as file:
                    existing = {line.strip() for line in file if line.strip()}

            if file_name not in existing:
                with open(BROKEN_FILES_PATH, "a", encoding="utf-8") as file:
                    if existing:
                        file.write("\n")
                    file.write(file_name)
                Logger.log("INFO", f"Added broken file to list: {file_name}")
        except Exception as e:
            Logger.log("WARNING", f"Failed to update broken files list for {file_name}: {e}")

    def _parse_region_and_id(self, img_path: str) -> Tuple[str, str]:
        region = os.path.basename(os.path.dirname(img_path))
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        return region, img_id

    def _calculate_sigmoid_params(self, x1, y1, x2, y2):
        """Calculate the parameters for the sigmoid function based on two points."""
        logit1 = np.log(y1 / (1 - y1))
        logit2 = np.log(y2 / (1 - y2))

        k = (logit2 - logit1) / (x2 - x1)
        x0 = x1 - logit1 / k

        return {"k": k, "x0": x0}

    def _custom_sigmoid(self, x):
        """Apply a custom sigmoid function with the calculated parameters."""
        k = self.sigmoid_params["k"]
        x0 = self.sigmoid_params["x0"]
        return 1 / (1 + np.exp(-k * (x - x0)))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, json_path = self.files[idx]

        Logger.log("DEBUG", f"Loading image: {img_path} with JSON: {json_path}")
        try:
            image = Image.open(img_path).convert("RGB") # TODO: Consider HSV, may get better results
        except Exception as e:
            Logger.log("ERROR", f"Failed to load image {img_path}: {e}")
            self._append_broken_file(img_path)
            # Return None and skip
            return None, None
        if self.transform:
            image = self.transform(image)

        label_vector = torch.zeros(len(self.salient_regions), dtype=torch.float32)

        if json_path:  # non-salient images do not have a JSON file, label defaults to zero
            with open(json_path, "r") as f:
                region_counts = json.load(f)
            total_count = sum(region_counts.values())
            for mgrs_zone, count in region_counts.items():
                if mgrs_zone in self.salient_region_indices:
                    i = self.salient_region_indices[mgrs_zone]
                    raw_value = count / total_count if total_count > 0 else 0
                    label_vector[i] = self._custom_sigmoid(raw_value)

        return image, label_vector
