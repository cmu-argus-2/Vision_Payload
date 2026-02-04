"""
Region Classification Module

This module defines the RegionClassifier class, which leverages a pretrained EfficientNet model to classify
images based on geographic regions. The classifier is tailored to recognize specific regions by adjusting the
final layer to match the number of target classes and loading custom model weights. Main functionalities
include image preprocessing and the execution of classification, providing class probabilities for each
recognized region.


Author: Eddie
Date: [Creation or Last Update Date]
"""

import os
from collections import defaultdict
from time import perf_counter
from typing import List

import cv2
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from utils.config_utils import USER_CONFIG_PATH, load_config
from vision_inference.frame import Frame
from vision_inference.logger import Logger
from vision_inference.numpy_dataloader import ImageSimInference


class RegionClassifier:
    """
    A class to classify MGRS regions in images using a pretrained EfficientNet model.
    """

    NUM_CLASSES = 40
    CONFIDENCE_THRESHOLD = 0.55
    DOWNSAMPLED_SIZE = (224, 224)
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    MODEL_WEIGHTS_RELATIVE_PATH = "rc_model_weights.pth"

    def __init__(self, load_weights: bool = True):
        """
        Initialize the RegionClassifier.

        Args:
            load_weights (bool): Whether to load model weights. Default is True.
        """
        Logger.log("INFO", "Initializing RegionClassifier.")

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ClassifierEfficient().to(self.device)

            # Load Custom model weights if required
            if load_weights:
                self.model.load_state_dict(
                    torch.load(RegionClassifier.get_model_weights_path(), map_location=self.device, weights_only=False)
                )
                self.model.eval()
                Logger.log("INFO", "Model loaded successfully.")

        except Exception as e:
            Logger.log("ERROR", f"Failed to load model: {e}")
            raise

        # Define the preprocessing
        self.transforms = transforms.Compose(
            [
                transforms.Resize(RegionClassifier.DOWNSAMPLED_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=RegionClassifier.IMAGE_NET_MEAN, std=RegionClassifier.IMAGE_NET_STD
                ),
            ]
        )

        self.region_ids = RegionClassifier.load_region_ids()

    @staticmethod
    def get_model_weights_path() -> str:
        """
        Get the path to the model weights file.

        Returns:
            The path to the model weights file.
        """
        models_dir = load_config(USER_CONFIG_PATH)["models_directory"]
        return os.path.join(models_dir, RegionClassifier.MODEL_WEIGHTS_RELATIVE_PATH)

    @staticmethod
    def load_region_ids() -> List[str]:
        """
        Load region IDs from the configuration file.

        Returns:
            A list of region IDs.
        """
        try:
            config = load_config()
            region_ids = config["vision"]["salient_mgrs_region_ids"]
            assert (
                len(region_ids) == RegionClassifier.NUM_CLASSES
            ), "Incorrect number of region IDs."
            assert (
                len(set(region_ids)) == RegionClassifier.NUM_CLASSES
            ), "Duplicate region IDs detected."
            return region_ids
        except Exception as e:
            Logger.log("ERROR", f"Configuration error: {e}")
            raise

    def classify_region(self, frame_obj: Frame) -> List[str]:
        """
        Classify the regions present in the given frame.

        :param frame_obj: The frame object to classify.
        :return: A list of region IDs.
        """
        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_name} frame {frame_obj.frame_id}] Starting the classification process.",
        )
        try:
            img = self.transforms(Image.fromarray(frame_obj.image)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                start_time = perf_counter()
                probabilities = self.model(img)
                inference_time = perf_counter() - start_time

                predicted = (probabilities > RegionClassifier.CONFIDENCE_THRESHOLD).float()
                predicted_indices = predicted.nonzero(as_tuple=True)[1]
                predicted_region_ids = [self.region_ids[idx] for idx in predicted_indices]

        except Exception as e:
            Logger.log("ERROR", f"Classification process failed: {e}")
            raise

        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_name} frame {frame_obj.frame_id}] {predicted_region_ids} region(s) identified.",
        )
        Logger.log("INFO", f"Inference completed in {inference_time:.2f} seconds.")
        return predicted_region_ids

    def _prepare_batch_data(self, images_dir: str, num_workers: int = 0) -> None:
        """
        Prepares the data loader for batch image classification.

        Args:
            images_dir (str): Directory containing images to classify
            num_workers (int): Number of worker processes to use for data loading.
                              Higher values may improve performance on multi-core systems.

        Returns:
            None: Sets up the dataset and dataloader attributes for the class.
        """
        self.images_dir = images_dir
        self.dataset = ImageSimInference(images_dir=images_dir, transform=self.transforms)
        self.dataloader = DataLoader(
            self.dataset, batch_size=16, shuffle=False, num_workers=num_workers
        )

    def classify_region_batch(
        self, images_dir: str, num_workers: int = 0
    ) -> tuple[dict[str, List[str]], dict[str, List[str]]]:
        """
        Classify regions in a batch of images from a directory.

        Args:
            images_dir (str): Directory containing images to classify
            num_workers (int): Number of worker processes for data loading

        Returns:
            tuple: Two dictionaries containing the classification results:
                - reg2img (dict[str, List[str]]): Mapping from region IDs to lists of image filenames
                - img2reg (dict[str, List[str]]): Mapping from image filenames to lists of region IDs

        Raises:
            Exception: If the batch classification process fails
        """
        Logger.log("INFO", f"Starting batch classification of images from {images_dir}")

        try:
            # Prepare data loader
            self._prepare_batch_data(images_dir, num_workers)

            img2reg = defaultdict(list)
            reg2img = defaultdict(list)
            batch_start_time = perf_counter()

            # Import tqdm for progress tracking
            from tqdm import tqdm
            total_batches = len(self.dataloader)
            
            # Process each batch with tqdm progress bar
            with torch.no_grad():
                for batch_idx, (images, paths) in tqdm(
                    enumerate(self.dataloader), 
                    total=total_batches,
                    desc="Classifying regions",
                    unit="batch"
                ):
                    images = images.to(self.device)

                    # Forward pass
                    probabilities = self.model(images)
                    predicted = (probabilities > RegionClassifier.CONFIDENCE_THRESHOLD).float()

                    # Extract predictions for each image in batch
                    for i in range(images.size(0)):
                        img_predicted_indices = predicted[i].nonzero(as_tuple=True)[0]
                        img_predicted_regions = [
                            self.region_ids[idx.item()] for idx in img_predicted_indices
                        ]
                        image_name = os.path.basename(paths[i])
                        for region in img_predicted_regions:
                            img2reg[image_name].append(region)
                            reg2img[region].append(image_name)

            total_time = perf_counter() - batch_start_time
            Logger.log(
                "INFO",
                f"Batch classification completed. Processed {len(self.dataset)} images, and classified {len(img2reg)} of them in {total_time:.2f} seconds",
            )
            return reg2img, img2reg

        except Exception as e:
            Logger.log("ERROR", f"Batch classification failed: {e}")
            raise


class ClassifierEfficient(nn.Module):
    """
    A custom classifier using the EfficientNet model.
    """

    def __init__(self):
        """
        Initialize the classifier.
        """
        super().__init__()
        # Using new weights system
        # This uses the most up-to-date weights
        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)
        for param in self.efficientnet.features[:3].parameters():
            param.requires_grad = False
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, RegionClassifier.NUM_CLASSES)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        :param x: The input tensor, representing an image.
        :return: The output tensor, representing a probability between 0 and 1 for each salient MGRS region.
        """
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x
