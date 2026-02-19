"""
VRAM-Resident Dataset Module for Full Dataset GPU Loading

This module provides a dataset class that loads the ENTIRE dataset into VRAM upfront,
then samples batches directly from GPU memory during training. This eliminates all
CPU-to-GPU transfer overhead during training at the cost of higher initial memory usage.

Use this when:
- Your entire dataset fits in VRAM
- You want zero CPU-GPU transfer overhead during training
- You have sufficient GPU memory (typically 8GB+ VRAM)

Classes:
    MGRSImageDatasetVRAM: Dataset that pre-loads all data into VRAM
    VRAMBatchSampler: Custom sampler for efficient VRAM batch sampling
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
from tqdm import tqdm

from utils.config_utils import BROKEN_FILES_PATH
from vision_inference.logger import Logger


class MGRSImageDatasetVRAM(Dataset):
    """
    A VRAM-resident dataset that pre-loads all images and labels into GPU memory.
    
    This dataset loads all data into VRAM during initialization, then returns
    pre-loaded GPU tensors during training. This completely eliminates CPU-GPU
    data transfer overhead during training.
    
    Memory Requirements:
    - For N images of size 3x224x224: ~0.6 MB per image
    - Example: 10,000 images = ~6 GB VRAM
    """

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
        device: Optional[torch.device] = None,
        preload_to_vram: bool = True,
        data_subset_percent: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        """
        Args:
            root_dir (str): Path to the dataset directory.
            broken_files (List[str]): List of broken file names to skip.
            root_dir_non_salient (Optional[str]): Path to the non-salient dataset directory.
            salient_regions (List[str]): List of MGRS regions to consider.
            transform (Optional[object]): Optional transforms to apply to images.
            split (str): One of 'train', 'val', or 'test'.
            train_ratio (float): Ratio of data to use for training.
            val_ratio (float): Ratio of data to use for validation.
            seed (int): Random seed for reproducibility.
            device (torch.device): Device to load data to (default: cuda if available).
            preload_to_vram (bool): Whether to pre-load all data to VRAM (default: True).
            data_subset_percent (float): Percentage of data to use (0.0-1.0, default: 1.0 for all data).
            num_workers (int): Number of workers for parallel data loading during initialization (default: 0).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.salient_regions = sorted(salient_regions or [])
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preload_to_vram = preload_to_vram and torch.cuda.is_available()
        self.num_workers = num_workers

        # Create mapping from region to index
        self.salient_region_indices = {region: i for i, region in enumerate(self.salient_regions)}

        # Set sigmoid parameters
        self.sigmoid_params = self._calculate_sigmoid_params(0.2, 0.05, 0.3, 0.95)

        Logger.log("INFO", f"[VRAM-Resident] Sigmoid parameters: k={self.sigmoid_params['k']:.4f}, x0={self.sigmoid_params['x0']:.4f}")

        # Collect images and their corresponding json files
        self.files = []
        salient_regions_set = set(self.salient_regions)
        broken_file_set = set(broken_files or [])
        
        Logger.log("INFO", f"[VRAM-Resident] Scanning directory: {root_dir}")
        for f in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, f)) and f in salient_regions_set:
                region_dir = os.path.join(root_dir, f)
                for file in os.listdir(region_dir):
                    if (file.endswith(".png") or file.endswith(".jpg")) and file not in broken_file_set:
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
        random.shuffle(self.files)

        total_size = len(self.files)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)

        if split == "train":
            self.files = self.files[:train_size]
        elif split == "val":
            self.files = self.files[train_size : train_size + val_size]
        else:  # test
            self.files = self.files[train_size + val_size :]

        # Apply subset percentage if specified (for debugging)
        if data_subset_percent < 1.0:
            subset_size = max(1, int(len(self.files) * data_subset_percent))
            self.files = self.files[:subset_size]
            Logger.log("INFO", f"[VRAM-Resident] Using {data_subset_percent*100:.1f}% of data: {subset_size} images")

        Logger.log("INFO", f"[VRAM-Resident] Total {split} images: {len(self.files)}")

        # VRAM Pre-loading: Load all data into GPU memory
        if self.preload_to_vram:
            self._preload_all_to_vram()
        else:
            Logger.log("WARNING", "[VRAM-Resident] VRAM pre-loading disabled, will load on-demand")
            self.vram_images = None
            self.vram_labels = None

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

    def _preload_all_to_vram(self):
        """
        Pre-load all images and labels into VRAM.
        This is done once during initialization to eliminate CPU-GPU transfer during training.
        Uses parallel workers for faster disk I/O if num_workers > 0.
        """
        Logger.log("INFO", f"[VRAM-Resident] Pre-loading {len(self.files)} images to VRAM (workers={self.num_workers})...")
        
        # Estimate memory requirements
        num_images = len(self.files)
        num_labels = len(self.salient_regions)
        # Assuming 3x224x224 float32 images: 3*224*224*4 bytes = ~600KB per image
        estimated_image_mem_mb = (num_images * 3 * 224 * 224 * 4) / (1024**2)
        estimated_label_mem_mb = (num_images * num_labels * 4) / (1024**2)
        estimated_total_mb = estimated_image_mem_mb + estimated_label_mem_mb
        
        Logger.log("INFO", f"[VRAM-Resident] Estimated VRAM usage: {estimated_total_mb:.1f} MB "
                  f"(images: {estimated_image_mem_mb:.1f} MB, labels: {estimated_label_mem_mb:.1f} MB)")
        
        # Check available GPU memory
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
            allocated_vram = torch.cuda.memory_allocated(self.device) / (1024**2)
            available_vram = total_vram - allocated_vram
            Logger.log("INFO", f"[VRAM-Resident] Available VRAM: {available_vram:.1f} MB / {total_vram:.1f} MB")
            
            if estimated_total_mb > available_vram * 0.8:  # Leave 20% buffer
                Logger.log("WARNING", f"[VRAM-Resident] Estimated memory usage ({estimated_total_mb:.1f} MB) "
                          f"may exceed available VRAM ({available_vram:.1f} MB). Consider reducing dataset size.")
        
        # Pre-allocate tensors on GPU
        self.vram_images = torch.empty((num_images, 3, 224, 224), dtype=torch.float32, device=self.device)
        self.vram_labels = torch.empty((num_images, num_labels), dtype=torch.float32, device=self.device)
        
        Logger.log("INFO", f"[VRAM-Resident] Allocated VRAM tensors: images={self.vram_images.shape}, labels={self.vram_labels.shape}")
        
        # Load all images and labels
        failed_count = 0
        for idx, (img_path, json_path) in enumerate(tqdm(self.files, desc="Loading to VRAM", unit="img")):
            try:
                # Load and transform image on CPU first
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image_tensor = self.transform(image)
                else:
                    from torchvision import transforms
                    image_tensor = transforms.ToTensor()(image)
                
                # Copy to pre-allocated GPU memory
                self.vram_images[idx].copy_(image_tensor, non_blocking=False)
                
                # Prepare label
                label_vector = torch.zeros(len(self.salient_regions), dtype=torch.float32)
                if json_path:
                    with open(json_path, "r") as f:
                        region_counts = json.load(f)
                    total_count = sum(region_counts.values())
                    for mgrs_zone, count in region_counts.items():
                        if mgrs_zone in self.salient_region_indices:
                            i = self.salient_region_indices[mgrs_zone]
                            raw_value = count / total_count if total_count > 0 else 0
                            label_vector[i] = self._custom_sigmoid(raw_value)
                
                # Copy label to GPU
                self.vram_labels[idx].copy_(label_vector, non_blocking=False)
                
            except Exception as e:
                Logger.log("ERROR", f"[VRAM-Resident] Failed to load {img_path}: {e}")
                failed_count += 1
                # Fill with zeros for failed images
                self.vram_images[idx].zero_()
                self.vram_labels[idx].zero_()
        
        if failed_count > 0:
            Logger.log("WARNING", f"[VRAM-Resident] Failed to load {failed_count}/{num_images} images")
        
        # Log final GPU memory usage
        if torch.cuda.is_available():
            allocated_vram = torch.cuda.memory_allocated(self.device) / (1024**2)
            reserved_vram = torch.cuda.memory_reserved(self.device) / (1024**2)
            Logger.log("INFO", f"[VRAM-Resident] GPU memory after loading: allocated={allocated_vram:.1f} MB, reserved={reserved_vram:.1f} MB")
        
        Logger.log("INFO", f"[VRAM-Resident] Successfully pre-loaded {len(self.files) - failed_count}/{len(self.files)} images to VRAM")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        If data is pre-loaded to VRAM, returns direct GPU tensor references.
        Otherwise, loads from disk (fallback mode).
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image_tensor, label_tensor), both on GPU if pre-loaded
        """
        if self.vram_images is not None:
            # Return pre-loaded GPU tensors directly (no copy, just reference)
            return self.vram_images[idx], self.vram_labels[idx]
        else:
            # Fallback: load from disk (slower)
            return self._load_from_disk(idx)
    
    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback method to load data from disk if not pre-loaded to VRAM."""
        img_path, json_path = self.files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            Logger.log("ERROR", f"[VRAM-Resident] Failed to load image {img_path}: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)

        label_vector = torch.zeros(len(self.salient_regions), dtype=torch.float32)

        if json_path:
            with open(json_path, "r") as f:
                region_counts = json.load(f)
            total_count = sum(region_counts.values())
            for mgrs_zone, count in region_counts.items():
                if mgrs_zone in self.salient_region_indices:
                    i = self.salient_region_indices[mgrs_zone]
                    raw_value = count / total_count if total_count > 0 else 0
                    label_vector[i] = self._custom_sigmoid(raw_value)

        return image, label_vector


class VRAMBatchSampler:
    """
    Custom batch sampler for VRAM-resident datasets.
    
    This sampler generates batch indices for efficient sampling from VRAM.
    It supports shuffling per epoch while maintaining contiguous memory access patterns.
    """
    
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = True, seed: int = 42):
        """
        Args:
            dataset_size: Total number of samples in the dataset
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices each epoch
            seed: Random seed for reproducibility
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
    def __iter__(self):
        """Generate batch indices for one epoch."""
        if self.shuffle:
            # Shuffle indices for this epoch
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(self.dataset_size)
        else:
            indices = np.arange(self.dataset_size)
        
        # Generate batches
        for start_idx in range(0, self.dataset_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.dataset_size)
            yield indices[start_idx:end_idx].tolist()
    
    def __len__(self):
        """Return number of batches per epoch."""
        return (self.dataset_size + self.batch_size - 1) // self.batch_size
    
    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling (changes random state)."""
        self.epoch = epoch
