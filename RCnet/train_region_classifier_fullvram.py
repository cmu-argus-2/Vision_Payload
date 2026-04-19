"""
Image Classification Module using EfficientNet - Full VRAM Loading

This module defines the `TrainRegionClassifierFullVRAM` class that loads the ENTIRE
dataset into VRAM upfront, then samples batches directly from GPU memory during training.

This approach:
- Eliminates ALL CPU-GPU data transfer overhead during training
- Provides fastest possible training speed
- Requires sufficient VRAM to hold the entire dataset
- Best for datasets that fit in GPU memory (typically <8GB for most GPUs)

Key Differences from Regular VRAM Mode:
- Regular VRAM: Streams data with pinned memory (still has transfer overhead)
- Full VRAM: Loads everything once, zero transfer overhead during training
"""

import gc
import os
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from dataloader_vram import MGRSImageDatasetVRAM, VRAMBatchSampler
from plotter import Plotter
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from utils.config_utils import USER_CONFIG_PATH, BROKEN_FILES_PATH, load_config

import wandb
from vision_inference.region_classifier import RegionClassifier as BaseRegionClassifier
from vision_inference.logger import Logger

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class TrainRegionClassifierFullVRAM(BaseRegionClassifier):
    """
    A deep learning-based multi-label image classifier with FULL VRAM pre-loading.
    
    This version loads the entire dataset into VRAM during initialization, then
    samples batches directly from GPU memory. This provides maximum training speed
    at the cost of higher VRAM usage.
    
    Memory Requirements:
    - ~0.6 MB per 224x224 RGB image
    - Example: 10,000 images ≈ 6 GB VRAM
    
    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model runs.
        train_dataset_vram (MGRSImageDatasetVRAM): VRAM-resident training dataset.
        val_dataset_vram (MGRSImageDatasetVRAM): VRAM-resident validation dataset.
        test_dataset_vram (MGRSImageDatasetVRAM): VRAM-resident test dataset.
        model (torch.nn.Module): The deep learning model for classification.
        plotter (Plotter): Utility for plotting training loss.
    """

    def __init__(
        self,
        data_path: str,
        broken_files_path: Optional[str] = None,
        non_salient_data_path: Optional[str] = None,
        selected_classes: Optional[List[str]] = None,
        save_plot_flag: bool = False,
        save_plot_path: Optional[str] = None,
        batch_size: int = 128,
        data_subset_percent: float = 1.0,
        num_workers: int = 0,
        vram_cache_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the TrainRegionClassifierFullVRAM and pre-loads all data to VRAM.

        Args:
            data_path (str): Path to the dataset.
            broken_files_path (str): Path to broken files configuration.
            non_salient_data_path (str): Path to non-salient data.
            selected_classes (list): List of selected classes for the multi-label classification.
            save_plot_flag (bool): Whether to save training loss plots.
            save_plot_path (str): Path to save the loss plot.
            batch_size (int): Batch size for training (default: 128).
            data_subset_percent (float): Percentage of data to use (default: 1.0 for 100%).
            num_workers (int): Number of workers for parallel data loading (default: 0).
            vram_cache_path (Optional[str]): Path to directory for caching VRAM tensors (default: None, uses ./vram_cache).
        """
        # Initializing Full-VRAM trainer
        
        # Initialize device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_subset_percent = data_subset_percent
        self.num_workers = num_workers
        self.vram_cache_path = vram_cache_path or "./vram_cache"

        # Prepare data and PRE-LOAD TO VRAM
        self._prepare_full_vram_data(
            data_path, broken_files_path, non_salient_data_path, selected_classes, batch_size, self.vram_cache_path
        )

        # Now initialize the parent class with our number of classes and skip weight loading
        assert len(self.regions) == BaseRegionClassifier.NUM_CLASSES, "Number of classes mismatch!"
        super().__init__(load_weights=False)

        # Initialize training specific components
        self.plotter = Plotter()
        self.save_plot_flag = save_plot_flag
        self.save_plot_path = save_plot_path
        self.batch_size = batch_size

    def _prepare_full_vram_data(
        self,
        data_path: str,
        broken_files: Optional[List[str]],
        non_salient_data_path: Optional[str],
        selected_classes: Optional[List[str]],
        batch_size: int,
        vram_cache_path: Optional[str] = None,
    ) -> None:
        """
        Prepares datasets by LOADING ALL DATA INTO VRAM upfront.
        
        This method:
        1. Loads all images from disk (or from cache)
        2. Applies transformations
        3. Stores everything in VRAM
        4. Creates samplers for efficient batch access

        Args:
            data_path (str): Path to the dataset directory.
            broken_files (list) (Optional): List of broken files.
            non_salient_data_path (str) (Optional): Path to non-salient data.
            selected_classes (list) (Optional): List of salient regions for classification.
            batch_size (int): Batch size for data loaders.
            vram_cache_path (Optional[str]): Path to directory for caching VRAM tensors.
        """
        if selected_classes is None:
            try:
                selected_classes = BaseRegionClassifier.load_region_ids()
                pass  # Using regions from configuration
            except Exception as e:
                selected_classes = sorted(os.listdir(data_path + "/train"))
                Logger.log("WARNING", "[Full-VRAM] Failed to load regions from config, using all available classes!")
                Logger.log("ERROR", f"Error: {e}")

        if broken_files is None:
            try:
                from utils.config_utils import BROKEN_FILES_PATH, load_config
                config = load_config(BROKEN_FILES_PATH)
                broken_files = config.get("broken_files", [])
                pass  # Loaded broken files from configuration
            except Exception as e:
                Logger.log("WARNING", "[Full-VRAM] Failed to load broken files from config!")
                Logger.log("ERROR", f"Error: {e}")
                broken_files = []
        
        self.regions = selected_classes

        # Define transforms - AGGRESSIVE AUGMENTATION for better generalization
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = transforms.Compose( 
            [
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Create VRAM-resident datasets (this loads all data to GPU or from cache)
        self.train_dataset_vram = MGRSImageDatasetVRAM(
            data_path,
            broken_files,
            non_salient_data_path,
            selected_classes,
            transform=self.train_transform,
            split="train",
            device=self.device,
            preload_to_vram=True,  # KEY: Pre-load everything to VRAM
            data_subset_percent=self.data_subset_percent,
            num_workers=self.num_workers,
            vram_cache_path=vram_cache_path,
        )
        
        self.val_dataset_vram = MGRSImageDatasetVRAM(
            data_path,
            broken_files,
            non_salient_data_path,
            selected_classes,
            transform=self.test_transform,
            split="val",
            device=self.device,
            preload_to_vram=True,  # KEY: Pre-load everything to VRAM
            data_subset_percent=self.data_subset_percent,
            num_workers=self.num_workers,
            vram_cache_path=vram_cache_path,
        )
        
        self.test_dataset_vram = MGRSImageDatasetVRAM(
            data_path,
            broken_files,
            non_salient_data_path,
            selected_classes,
            transform=self.test_transform,
            split="test",
            device=self.device,
            preload_to_vram=True,  # KEY: Pre-load everything to VRAM
            data_subset_percent=self.data_subset_percent,
            num_workers=self.num_workers,
            vram_cache_path=vram_cache_path,
        )

        # Create custom batch samplers
        self.train_sampler = VRAMBatchSampler(
            len(self.train_dataset_vram), batch_size, shuffle=True
        )
        self.val_sampler = VRAMBatchSampler(
            len(self.val_dataset_vram), batch_size, shuffle=False
        )
        self.test_sampler = VRAMBatchSampler(
            len(self.test_dataset_vram), batch_size, shuffle=False
        )

        # Datasets loaded to VRAM - now clean up any remaining CPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        Logger.log("INFO", "[Full-VRAM] Garbage collection completed after dataset loading")

    def _get_vram_batch(self, dataset, indices):
        """
        Get a batch directly from VRAM using indices.
        
        This is the key optimization: we're indexing pre-loaded GPU tensors,
        not transferring data from CPU to GPU.
        
        Args:
            dataset: VRAM-resident dataset
            indices: List of indices to sample
            
        Returns:
            Tuple of (images, labels) as GPU tensors
        """
        # Direct indexing into VRAM tensors (zero copy overhead!)
        images = dataset.vram_images[indices]
        labels = dataset.vram_labels[indices]
        return images, labels

    def train(self, epochs: int = 10, learning_rate: float = 1e-3) -> None:
        """
        Trains the model with ZERO CPU-GPU data transfer overhead.
        
        All data is already in VRAM, so training only involves:
        1. Indexing into VRAM tensors (extremely fast)
        2. Forward pass
        3. Backward pass
        4. Optimizer step
        
        No data loading, no transfers, maximum speed!

        Args:
            epochs (int): Number of training epochs. Default is 10.
            learning_rate (float): Learning rate for optimization. Default is 1e-3.
        """
        wandb.init(
            project="RCnet-FullVRAM",
            config={
                "mode": "train",
                "epochs": epochs,
                "learning_rate": learning_rate,
                "architecture": "EfficientNet-b0",
                "dataset": "Landsat Mosaic",
                "optimization": "Full-VRAM-Resident",
                "batch_size": self.batch_size,
                "data_location": "VRAM",
                "cpu_gpu_transfers": "ZERO",
            },
        )
        wandb.watch(self.model, log="all", log_freq=100)
        #criterion = nn.BCEWithLogitsLoss()
        criterion = FocalLoss(alpha=float(os.environ.get('RC_ALPHA', '0.90')), gamma=int(os.environ.get('RC_GAMMA', '3')))
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )

        proc = psutil.Process(os.getpid())
        best_val_f1 = -1.0
        self.best_model_path = getattr(self, 'best_model_path', 'RCnet/output/model_best.pth')

        for epoch in range(epochs):
            # Update sampler epoch for shuffling
            self.train_sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            total_images = len(self.train_dataset_vram)
            processed_images = 0

            progress_bar = tqdm(
                self.train_sampler,
                desc=f"[Full-VRAM] Epoch {epoch + 1}/{epochs}",
                leave=True,
                unit="batch",
                total=len(self.train_sampler),
            )

            mem_start_epoch = proc.memory_info().rss / 1024**2
            gpu_mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            
            wandb.log({
                "epoch_start_memory_MB": mem_start_epoch,
                "epoch_start_gpu_allocated_MB": gpu_mem_allocated,
                "epoch_start_gpu_reserved_MB": gpu_mem_reserved
            })

            for batch_idx, indices in enumerate(progress_bar):
                # KEY OPTIMIZATION: Get batch directly from VRAM (zero copy!)
                data, targets = self._get_vram_batch(self.train_dataset_vram, indices)
                
                batch_size_actual = data.size(0)
                processed_images += batch_size_actual

                # Forward pass (data already on GPU!)
                scores = self.model(data)
                loss = criterion(scores, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(
                    {"imgs": f"{processed_images}/{total_images}", "loss": f"{loss.item():.4f}"}
                )

                wandb.log({
                    "batch_loss": loss.item(),
                }, commit=True)

                epoch_loss += loss.item() * batch_size_actual

            epoch_loss /= total_images
            mem_end_epoch = proc.memory_info().rss / 1024**2
            
            # Clean up memory after epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            wandb.log({
                "epoch": epoch+1,
                "epoch_loss": epoch_loss,
                "epoch_end_memory_MB": mem_end_epoch
            }, commit=True)
            
            self.plotter.update_loss(epoch_loss)

            # Save checkpoint
            self.save_model(path=f"RCnet/chkpts/model_fullvram_{epoch + 1}.pth")
            val_f1 = self.validate()
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr})

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model(path=self.best_model_path)
                print(f"[BEST] Epoch {epoch+1}: F1={val_f1:.2f} saved to {self.best_model_path}")

            # Clean up after validation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if epoch == epochs - 1:
                test_accuracy = self.evaluate()
                wandb.log({"test_accuracy": test_accuracy})

        if self.save_plot_flag:
            self.plotter.save_plot(self.save_plot_path)
            wandb.log({"loss_vs_epoch": wandb.Image(self.save_plot_path)})

    def validate(self) -> float:
        """
        Validates model with zero data transfer overhead.
        
        Returns:
            float: Validation F1 score in percentage.
        """
        self.model.eval()

        # Track TP/FP/FN under both label conventions:
        #   STRICT = labels > 0.25 (only dominant regions count as ground-truth positive)
        #   SOFT   = labels != 0   (any coverage counts as ground-truth positive)
        counts = {'strict': [0, 0, 0], 'soft': [0, 0, 0]}  # tp, fp, fn

        with torch.no_grad():
            progress_bar = tqdm(self.val_sampler, desc="[Full-VRAM] Validating", leave=True, unit="batch")
            for indices in progress_bar:
                images, labels = self._get_vram_batch(self.val_dataset_vram, indices)
                outputs = self.model(images)
                predictions = outputs > 0.5

                for name, label_bin in [('strict', labels > 0.25), ('soft', labels.bool())]:
                    counts[name][0] += (predictions & label_bin).sum().item()
                    counts[name][1] += (predictions & ~label_bin).sum().item()
                    counts[name][2] += (~predictions & label_bin).sum().item()

        metrics = {}
        for name in ('strict', 'soft'):
            tp, fp, fn = counts[name]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics[f'validation_f1_{name}'] = f1 * 100
            metrics[f'validation_precision_{name}'] = p * 100
            metrics[f'validation_recall_{name}'] = r * 100

        # Also keep original keys (= STRICT) for plotter / best-model selection
        metrics['validation_f1_score'] = metrics['validation_f1_strict']
        metrics['validation_precision'] = metrics['validation_precision_strict']
        metrics['validation_recall'] = metrics['validation_recall_strict']
        wandb.log(metrics)

        precision = metrics['validation_precision_strict'] / 100
        recall = metrics['validation_recall_strict'] / 100
        f1_score = metrics['validation_f1_strict'] / 100
        
        # Clean up after validation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return f1_score * 100

    def evaluate(self, output_file: str = "RCnet/results/evaluation_results_fullvram.txt") -> float:
        """
        Evaluates model on test set with zero data transfer overhead.

        Args:
            output_file (str): File path to save evaluation results.

        Returns:
            float: Test F1 score in percentage.
        """
        if wandb.run is None:
            wandb.init(
                project="RCnet-FullVRAM",
                config={
                    "mode": "eval",
                    "architecture": "EfficientNet-b0",
                    "dataset": "Sentinel",
                    "optimization": "Full-VRAM-Resident",
                },
            )

        self.model.eval()
        all_features = []
        all_labels = []
        class_correct = {i: 0 for i in range(len(self.regions))}
        class_total = {i: 0 for i in range(len(self.regions))}
        class_images = {i: [] for i in range(len(self.regions))}
        
        tot_time = 0
        
        with torch.no_grad():
            for indices in tqdm(self.test_sampler, desc="[Full-VRAM] Evaluating", leave=False):
                # Get batch directly from VRAM
                images, labels = self._get_vram_batch(self.test_dataset_vram, indices)

                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()
                predicted = outputs > 0.5
                tot_time += end_time - start_time

                # Store features and labels
                all_features.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Group images by predicted classes
                for i in range(images.size(0)):
                    predicted_classes = [j for j, val in enumerate(predicted[i]) if val]
                    for pred_class in predicted_classes:
                        if pred_class < len(class_images) and len(class_images[pred_class]) < 16:
                            class_images[pred_class].append(images[i].cpu())

                # Compute per-class accuracy (STRICT: only regions with label > 0.5)
                for class_idx in range(labels.size(1)):
                    class_labels = labels[:, class_idx] > 0.25
                    class_preds = predicted[:, class_idx]
                    class_correct[class_idx] += (class_labels & class_preds).sum().item()
                    class_total[class_idx] += class_labels.sum().item()

        # Concatenate all features and labels
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate final metrics (STRICT target definition: label > 0.5 = positive)
        predicted_all = (torch.tensor(all_features) > 0.5).bool()
        labels_all = torch.tensor(all_labels) > 0.25
        
        true_positives = (predicted_all & labels_all).sum().item()
        false_positives = (predicted_all & ~labels_all).sum().item()
        false_negatives = (~predicted_all & labels_all).sum().item()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Compute class-wise accuracies
        class_accuracies = {
            self.regions[class_idx]: (
                (100 * class_correct[class_idx] / class_total[class_idx])
                if class_total[class_idx] > 0
                else 0
            )
            for class_idx in class_correct
        }

        # Plot and save results
        plt.figure(figsize=(12, 6))
        plt.bar(class_accuracies.keys(), class_accuracies.values(), color="blue")
        plt.xlabel("Class Index")
        plt.ylabel("Accuracy (%)")
        plt.title("Class-wise Accuracies (Full VRAM)")
        plt.xticks(ticks=range(len(self.regions)), labels=self.regions, rotation=90)
        plt.ylim(0, 100)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plot_path = os.path.join(os.path.dirname(output_file), "class_wise_accuracies_fullvram.png")
        plt.savefig(plot_path)
        plt.close()
        
        wandb.log({"class_wise_accuracies_plot": wandb.Image(plot_path)})
        wandb.log({
            "overall_f1_score": f1_score * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            **{f"{k}_accuracy": v for k, v in class_accuracies.items()},
        })

        # Evaluation complete - clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return f1_score * 100

    def save_model(self, path: str = "model.pth") -> None:
        """Saves the trained model."""
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create directory if path contains one
            os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "model.pth") -> None:
        """Loads a saved model."""
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
