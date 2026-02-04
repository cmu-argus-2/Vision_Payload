"""
Image Batch Processing Module for ML Inference

This module provides datasets and data loading utilities for efficient batch processing 
of image files (.png) in machine learning inference tasks.

Key features:
- Brightness-based image filtering to exclude under-exposed frames
- PyTorch Dataset implementation for integration with DataLoader
- Configurable transformation pipelines for image preprocessing
- Robust error handling for corrupt or invalid image files
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Callable, Optional
import cv2
from PIL import Image

from vision_inference.logger import Logger


class ImageSimInference(Dataset):
    """
    Dataset class for processing image files in batches.
    Includes brightness filtering similar to frame_processor logic.
    """
    
    def __init__(
        self, 
        images_dir: str,
        transform: Optional[Callable] = None,
        file_extension: str = "*.png",
        brightness_filter: bool = True,
        dark_threshold: float = 0.9,
        brightness_threshold: int = 60
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory path containing image files
            transform: Transformation to apply to images
            file_extension: File extension pattern to match (default: "*.png")
            brightness_filter: Whether to apply brightness filtering (default: True)
            dark_threshold: Maximum percentage of dark pixels allowed for valid images (default: 0.9)
            brightness_threshold: Pixel value below which pixels are considered dark (default: 60)
        """
        self.images_dir = images_dir
        self.transform = transform
        self.dark_threshold = dark_threshold
        self.brightness_threshold = brightness_threshold
        
        # Find all matching files in the directory
        all_image_paths = sorted(glob.glob(os.path.join(images_dir, file_extension)))
        
        if not all_image_paths:
            Logger.log("WARNING", f"No files matching '{file_extension}' found in {images_dir}")
            self.image_paths = []
            return
            
        # Filter images based on brightness if enabled
        if brightness_filter:
            self.image_paths = []
            skipped_count = 0
            
            for path in all_image_paths:
                try:
                    # Load and check brightness
                    img = cv2.imread(path)
                    if img is None:
                        raise ValueError(f"Failed to load image: {path}")
                    if self._is_image_bright_enough(img):
                        self.image_paths.append(path)
                    else:
                        skipped_count += 1
                except Exception as e:
                    Logger.log("ERROR", f"Error checking brightness for {path}: {e}")
                    skipped_count += 1
                    
            Logger.log("INFO", f"Loaded {len(self.image_paths)} images, skipped {skipped_count} dim images")
        else:
            # Use all images without filtering
            self.image_paths = all_image_paths
            Logger.log("INFO", f"Loaded all {len(self.image_paths)} images without brightness filtering")
    
    def _is_image_bright_enough(self, image: np.ndarray) -> bool:
        """
        Check if image meets brightness criteria for ML processing.
        
        Args:
            image: NumPy array containing image data
            
        Returns:
            Boolean indicating if image is bright enough
        """
        try:
            if len(image.shape) == 3:
                gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = image  # Already grayscale
                
            dark_percentage = np.sum(gray_frame < self.brightness_threshold) / np.prod(gray_frame.shape)
            return dark_percentage <= self.dark_threshold
        except Exception as e:
            Logger.log("ERROR", f"Error in brightness calculation: {e}")
            return False  # Reject images we can't process
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (processed_image, image_path)
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load image file
            if image_path.endswith('.npy') or image_path.endswith('.npz'):
                image = np.load(image_path)
            else:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Image {image_path} could not be loaded.")

            # Convert to PIL Image for transforms
            # OpenCV uses BGR format, convert to RGB for PIL
            if image.shape[2] == 3:  # Check if it has color channels
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            pil_image = Image.fromarray(image)
            if pil_image.mode != 'RGB':
                Logger.log("WARNING", f"Image {image_path} is not RGB, converting to RGB")
                pil_image = pil_image.convert('RGB')

            # Apply transformations if provided
            if self.transform:
                image_tensor = self.transform(pil_image)
            else:
                # Basic conversion to tensor if no transform is provided
                from torchvision import transforms
                image_tensor = transforms.ToTensor()(pil_image)
            return image_tensor, image_path
            
        except Exception as e:
            Logger.log("ERROR", f"Error loading image {image_path}: {e}")
            # Return a placeholder tensor and the path
            placeholder = torch.zeros((3, 224, 224))
            if self.transform:
                # Try to match the expected tensor format from transform
                try:
                    dummy_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
                    placeholder = self.transform(dummy_img)
                except:
                    pass
            
            return placeholder, image_path


