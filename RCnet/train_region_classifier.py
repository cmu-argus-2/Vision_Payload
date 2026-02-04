"""
Image Classification Module using EfficientNet.

This module defines the `TrainRegionClassifier` class for training, 
evaluating, and validating an image classification model.  
It leverages EfficientNet-B0 as the backbone, supports logging with Weights & Biases (wandb),  
and provides utilities for dataset preparation, training, and performance evaluation.
"""

import os
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from data_loader import MGRSImageDataset as ImageDataset
from plotter import Plotter
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import wandb
from vision_inference.region_classifier import RegionClassifier as BaseRegionClassifier
from vision_inference.logger import Logger


class TrainRegionClassifier(BaseRegionClassifier):
    """
    A deep learning-based multi-label image classifier using EfficientNet.

    This class provides methods for preparing datasets, training, validation,
    evaluation, and saving/loading models. It integrates with Weights & Biases (wandb)
    for logging and visualization.

    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model runs.
        train_loader (DataLoader): DataLoader for training dataset.
        test_loader (DataLoader): DataLoader for testing dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        model (torch.nn.Module): The deep learning model for classification.
        plotter (Plotter): Utility for plotting training loss.
        save_plot_flag (bool): Flag to determine whether to save training loss plots.
        save_plot_path (str): Path to save the loss plot.

    Methods:
        _prepare_data(data_path):
            Loads and preprocesses the dataset into DataLoaders.

        _initialize_model():
            Initializes the EfficientNet-B0 model with a modified classifier layer.

        train(epochs=10, learning_rate=1e-3):
            Trains the model and logs loss/accuracy using wandb.

        validate():
            Evaluates model performance on the validation dataset.

        evaluate(output_file="RCnet/results/evaluation_results.txt"):
            Evaluates the model on the test dataset and logs results.

        save_model(path="model.pth"):
            Saves the trained model's weights.

        load_model(path="model.pth"):
            Loads a saved model's weights and sets it to evaluation mode.
    """

    def __init__(
        self,
        data_path: str,
        non_salient_data_path: Optional[str] = None,
        selected_classes: Optional[List[str]] = None,
        save_plot_flag: bool = False,
        save_plot_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the TrainRegionClassifier for training.

        Args:
            data_path (str): Path to the dataset.
            selected_classes (list): List of selected classes for the multi-label classification.
            save_plot_flag (bool): Whether to save training loss plots.
            save_plot_path (str): Path to save the loss plot.
        """
        # Prepare data first to get the number of classes
        self._prepare_batch_data(data_path, non_salient_data_path, selected_classes)

        # Now initialize the parent class with our number of classes and skip weight loading
        assert len(self.regions) == BaseRegionClassifier.NUM_CLASSES, "Number of classes mismatch!"
        super().__init__(load_weights=False)

        # Initialize training specific components
        self.plotter = Plotter()
        self.save_plot_flag = save_plot_flag
        self.save_plot_path = save_plot_path

    def _prepare_batch_data(
        self,
        data_path: str,
        non_salient_data_path: Optional[str],
        selected_classes: Optional[List[str]],
    ) -> None:
        """
        Prepares the dataset by applying transformations and loading it into DataLoaders.

        Args:
            data_path (str): Path to the dataset directory.
            selected_classes (list): List of salient regions for classification.
        """
        if selected_classes is None:
            # Use all regions from configuration using the parent class's method
            try:
                selected_classes = BaseRegionClassifier.load_region_ids()
                Logger.log("INFO", f"Using {len(selected_classes)} regions from configuration")
            except Exception as e:
                # Fallback to directory scanning if config loading fails
                selected_classes = sorted(os.listdir(data_path + "/train"))
                Logger.log("WARNING", "Failed to load regions from config, using all available classes for training!")
                Logger.log("ERROR", f"Error: {e}")

        self.regions = selected_classes
        Logger.log("INFO", f"Using regions: {self.regions}")

        # Define transforms for training and testing sets
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                # transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ToTensor(),
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                # transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
            ]
        )

        # Load datasets with appropriate transforms
        train_dataset = ImageDataset(
            data_path,
            non_salient_data_path,
            selected_classes,
            transform=self.train_transform,
            split="train",
        )
        val_dataset = ImageDataset(
            data_path,
            non_salient_data_path,
            selected_classes,
            transform=self.test_transform,
            split="val",
        )
        test_dataset = ImageDataset(
            data_path,
            non_salient_data_path,
            selected_classes,
            transform=self.test_transform,
            split="test",
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=0,  # Do not use multiple workers, we hit I/O limits on the HDD
            pin_memory=False,
            # persistent_workers=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            # persistent_workers=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            # persistent_workers=True
        )

        Logger.log("INFO", f"Train dataset size: {len(train_dataset)}")
        Logger.log("INFO", f"Validation dataset size: {len(val_dataset)}")
        Logger.log("INFO", f"Test dataset size: {len(test_dataset)}")

    def train(self, epochs: int = 10, learning_rate: float = 1e-3) -> None:
        """
        Trains the image classifier using EfficientNet-B0 and logs progress using wandb.

        Args:
            epochs (int): Number of training epochs. Default is 10.
            learning_rate (float): Learning rate for optimization. Default is 1e-3.
        """
        wandb.init(
            project="RCnet",
            config={
                "mode": "train",
                "epochs": epochs,
                "learning_rate": learning_rate,
                "architecture": "EfficientNet-b0",
                "dataset": "Sentinel",
            },
        )
        wandb.watch(self.model, log="all", log_freq=100)

        # CHANGED: Use BCE loss instead of BCEWithLogitsLoss since model already applies sigmoid
        criterion = nn.BCELoss()  # BCE loss for multi-label classification
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        proc = psutil.Process(os.getpid())

        for epoch in range(epochs):
            epoch_loss = 0.0  # Initialize epoch loss
            total_images = len(self.train_loader.dataset)
            processed_images = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
                unit="batch",
                total=len(self.train_loader),
            )

            mem_start_epoch = proc.memory_info().rss / 1024**2  # in MB
            Logger.log("INFO", f"[Epoch {epoch+1}] start memory: {mem_start_epoch:.1f} MB")
            wandb.log({"epoch_start_memory_MB": mem_start_epoch})

            # pylint: disable=unused-variable
            for batch_idx, (data, targets) in enumerate(progress_bar):
                Logger.log("INFO", f"Completed batch: {batch_idx}")
                batch_size = data.size(0)
                processed_images += batch_size

                mem_before = proc.memory_info().rss / 1024**2

                data = data.to(self.device)
                targets = targets.to(self.device).float()  # Ensure targets are float for BCE
                scores = self.model(data)  # Model already applies sigmoid
                loss = criterion(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar with image count and loss
                progress_bar.set_postfix(
                    {"imgs": f"{processed_images}/{total_images}", "loss": f"{loss.item():.4f}"}
                )

                mem_after = proc.memory_info().rss / 1024**2
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_memory_before_MB": mem_before,
                    "batch_memory_after_MB": mem_after,
                    "batch_memory_delta_MB": mem_after - mem_before
                }, commit=True)

                epoch_loss += loss.item() * batch_size  # Accumulate batch loss


            epoch_loss /= total_images  # Compute average batch loss
            mem_end_epoch = proc.memory_info().rss / 1024**2
            Logger.log("INFO", f"Epoch [{epoch+1}/{epochs}] Avg Loss: {epoch_loss:.4f} | end memory: {mem_end_epoch:.1f} MB")
            wandb.log({
                "epoch": epoch+1,
                "epoch_loss": epoch_loss,
                "epoch_end_memory_MB": mem_end_epoch
            }, commit=True)
            self.plotter.update_loss(epoch_loss)

            # if epoch % 2 == 0:
            self.save_model(path="RCnet/chkpts/model" + str(epoch + 1) + ".pth")
            self.validate()
            if epoch == epochs - 1:
                test_accuracy = self.evaluate()
                wandb.log({"test_accuracy": test_accuracy})


        if self.save_plot_flag:
            self.plotter.save_plot(self.save_plot_path)
            wandb.log({"loss_vs_epoch": wandb.Image(self.save_plot_path)})

    def save_model(self, path: str = "model.pth") -> None:
        """
        Saves the trained model to the specified path.

        Args:
            path (str): Path to save the model file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "model.pth") -> None:
        """
        Loads the trained model from the specified path.

        Args:
            path (str): Path to the saved model file.
        """

        self.model.load_state_dict(torch.load(path, weights_only=True))

        self.model.eval()

    def validate(self) -> float:
        """
        Evaluates model performance on the validation dataset.

        Returns:
            float: Validation F1 score in percentage.
        """
        self.model.eval()  # Set the model to evaluation mode

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():  # No gradient is needed for validation
            progress_bar = tqdm(self.val_loader, desc="Validating", leave=True, unit="batch")
            for images, labels in progress_bar:  # Use the progress_bar instead of self.val_loader
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)  # Model already applies sigmoid
                # CHANGED: No need to apply sigmoid again
                predictions = outputs > 0.5  # Direct thresholding for multi-label

                # Calculate multi-label metrics
                true_positives += (predictions & labels.bool()).sum().item()
                false_positives += (predictions & ~labels.bool()).sum().item()
                false_negatives += (~predictions & labels.bool()).sum().item()

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Log metrics
        wandb.log(
            {
                "validation_f1_score": f1_score * 100,
                "validation_precision": precision * 100,
                "validation_recall": recall * 100,
            }
        )

        Logger.log("INFO", f"Validation F1 Score: {f1_score * 100:.2f}%")
        Logger.log("INFO", f"Validation Precision: {precision * 100:.2f}%")
        Logger.log("INFO", f"Validation Recall: {recall * 100:.2f}%")

        return f1_score * 100  # Return F1 score as percentage

    # pylint: disable=too-many-locals
    def evaluate(self, output_file: str = "RCnet/results/evaluation_results.txt") -> float:
        """
        Evaluates the model on the test dataset and logs results.

        Args:
            output_file (str): File path to save evaluation results.

        Returns:
            float: Test accuracy in percentage.
        """
        if wandb.run is None:
            wandb.init(
                project="RCnet",
                config={
                    "mode": "eval",
                    "architecture": "EfficientNet-b0",
                    "dataset": "Sentinel",
                },
            )

        self.model.eval()
        all_features = []
        all_labels = []
        class_correct = {i: 0 for i in range(len(self.regions))}
        class_total = {i: 0 for i in range(len(self.regions))}

        class_images = {i: [] for i in range(len(self.regions))}  # Store images per class
        tot_time = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                start_time = time.time()
                outputs = self.model(images)  # Model already applies sigmoid
                end_time = time.time()

                # CHANGED: No need to handle probabilities separately, outputs are already probabilities
                predicted = outputs > 0.5  # Direct thresholding for multi-label (keep as boolean)
                tot_time += end_time - start_time

                # Store features and labels for t-SNE
                all_features.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Group images by their predicted classes
                for i in range(images.size(0)):
                    predicted_classes = [j for j, val in enumerate(predicted[i]) if val]
                    for pred_class in predicted_classes:
                        if pred_class < len(class_images):
                            class_images[pred_class].append(images[i].cpu())

                # For sample-wise accuracy (exact matches)
                exact_matches = (
                    ((predicted == labels.bool()).sum(dim=1) == labels.size(1)).sum().item()
                )
                sample_accuracy = 100 * exact_matches / labels.size(0)

                # For label-wise metrics
                true_positives = (predicted & labels.bool()).sum().item()
                false_positives = (predicted & ~labels.bool()).sum().item()
                false_negatives = (~predicted & labels.bool()).sum().item()

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1_score = (
                    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                )

                # Calculate exact match ratio (all labels correct for each sample)
                exact_matches = (
                    ((predicted == labels.bool()).sum(dim=1) == labels.size(1)).sum().item()
                )
                exact_match_ratio = exact_matches / labels.size(0)

                # Compute per-class accuracy
                for class_idx in range(labels.size(1)):  # Iterate over classes
                    class_labels = labels[:, class_idx].bool()
                    class_preds = predicted[:, class_idx]

                    # CHANGED: Use boolean operations instead of float
                    class_correct[class_idx] += (class_labels & class_preds).sum().item()
                    class_total[class_idx] += class_labels.sum().item()

            # Convert collected features and labels to NumPy for t-SNE
            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # Log images per class
            for class_id, image_list in class_images.items():
                if len(image_list) > 0:
                    # CHANGED: Only take up to 16 images to avoid memory issues
                    sample_images = image_list[: min(16, len(image_list))]
                    if sample_images:
                        image_grid = torch.stack(sample_images, dim=0)
                        class_name = self.regions[class_id]
                        wandb.log(
                            {
                                f"class_{class_name}_images": wandb.Image(
                                    image_grid, caption=f"Class {class_name} predictions"
                                )
                            }
                        )

            # Compute class-wise accuracies
            class_accuracies = {
                self.regions[class_idx]: (
                    (100 * class_correct[class_idx] / class_total[class_idx])
                    if class_total[class_idx] > 0
                    else 0
                )
                for class_idx in class_correct
            }

            # Plot bar chart for class-wise accuracies
            plt.figure(figsize=(12, 6))
            plt.bar(class_accuracies.keys(), class_accuracies.values(), color="blue")
            plt.xlabel("Class Index")
            plt.ylabel("Accuracy (%)")
            plt.title("Class-wise Accuracies")
            plt.xticks(
                ticks=range(len(self.regions)), labels=self.regions, rotation=90
            )  # Assuming 40 classes
            plt.ylim(0, 100)  # Accuracy range 0-100%

            # Save the figure and log to wandb
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plot_path = os.path.join(os.path.dirname(output_file), "class_wise_accuracies.png")
            plt.savefig(plot_path)
            plt.close()
            wandb.log({"class_wise_accuracies_plot": wandb.Image(plot_path)})

            # Log overall accuracy and per-class accuracies
            wandb.log(
                {
                    "overall_f1_score": f1_score * 100,
                    "precision": precision * 100,
                    "recall": recall * 100,
                    "exact_match_ratio": exact_match_ratio * 100,
                    **{f"{k}_accuracy": v for k, v in class_accuracies.items()},
                }
            )

            Logger.log("INFO", f"F1 score of the network on the test images: {f1_score * 100:.2f}%")
            Logger.log("INFO", f"Exact match ratio: {exact_match_ratio * 100:.2f}%")
            Logger.log("INFO", f"Total Inf time:{tot_time}")
            # Generate binary predictions
            # all_preds = (all_features > 0.5).astype(int)

            # save labels and preds
            # data = np.load(os.path.join(os.path.dirname(output_file), "predictions.npz"))  # update this path if needed
            # all_preds = data["predictions"]
            # all_labels = data["labels"]

            # num_classes = all_labels.shape[1]
            # multi_label_cm = np.zeros((num_classes, num_classes), dtype=int)

            # # Iterate over all samples
            # for true_vec, pred_vec in zip(all_labels, all_preds):
            #     true_indices = np.where(true_vec == 1)[0]
            #     pred_indices = np.where(pred_vec == 1)[0]
            #     for ti in true_indices:
            #         for pi in pred_indices:
            #             multi_label_cm[ti, pi] += 1

            # fig, ax = plt.subplots(figsize=(12, 12))
            # disp = ConfusionMatrixDisplay(multi_label_cm, display_labels=self.regions)
            # disp.plot(ax=ax, xticks_rotation=90, cmap='Blues', colorbar=False)
            # ax.set_title("Multi-label Confusion Matrix")
            # summary_cm_path = os.path.join(os.path.dirname(output_file), "multi_label_confusion_matrix.png")
            # plt.savefig(summary_cm_path)
            # plt.close()

            # wandb.log({"multi_label_confusion_matrix": wandb.Image(summary_cm_path)})
            # cls_10s = self.regions.index("10S")
            # cls_10t = self.regions.index("10T")

            # def get_group(vec, cls_10s, cls_10t):
            #     has_10s = vec[cls_10s] == 1
            #     has_10t = vec[cls_10t] == 1
            #     if has_10s and has_10t:
            #         return 2  # both
            #     elif has_10s:
            #         return 0  # only 10S
            #     elif has_10t:
            #         return 1  # only 10T
            #     else:
            #         return 3  # other

            # # Ensure binary
            # all_preds = (all_preds > 0.5).astype(int)
            # all_labels = (all_labels > 0.5).astype(int)

            # # Initialize 4x4 matrix
            # matrix_4x4 = np.zeros((4, 4), dtype=int)

            # for true_vec, pred_vec in zip(all_labels, all_preds):
            #     gt_group = get_group(true_vec, cls_10s, cls_10t)
            #     pred_group = get_group(pred_vec, cls_10s, cls_10t)
            #     matrix_4x4[gt_group, pred_group] += 1

            # row_labels = ["GT: only 10S", "GT: only 10T", "GT: both", "GT: other"]
            # col_labels = ["Pred: only 10S", "Pred: only 10T", "Pred: both", "Pred: other"]

            # fig, ax = plt.subplots(figsize=(8, 6))
            # im = ax.imshow(matrix_4x4, cmap='Blues')

            # # Annotate
            # for i in range(4):
            #     for j in range(4):
            #         ax.text(j, i, matrix_4x4[i, j], ha="center", va="center", color="black")

            # ax.set_xticks(np.arange(4))
            # ax.set_yticks(np.arange(4))
            # ax.set_xticklabels(col_labels, rotation=45, ha="right")
            # ax.set_yticklabels(row_labels)
            # ax.set_title("Custom 4x4 Confusion Matrix (10S vs 10T)")
            # plt.tight_layout()
            # plt.savefig("RCnet/results/custom_10s_10t_matrix.png")
            # plt.show()
            # wandb.log({"custom_10S_10T_matrix": wandb.Image("RCnet/results/custom_10s_10t_matrix.png")})
            # # return f1_score * 100
            # return 0.0  # Placeholder for actual accuracy calculation
