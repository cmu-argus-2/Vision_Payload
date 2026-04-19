"""
Main script for the RCnet training module.

This script initializes and runs the training pipeline
"""

import argparse

import torch

from train_region_classifier import TrainRegionClassifier
from train_region_classifier_fullvram import TrainRegionClassifierFullVRAM


def parse_args():
    """
    Parse command-line arguments for training or evaluating the image classifier.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Command-line Arguments:
        --train_flag (bool): Enable training mode if set.
        --save_plot_flag (bool): Save training plots if set.
        --vram (bool): Load entire dataset to VRAM for maximum speed.
        --data_dir (str, required): Path to the dataset directory.
        --non_salient_data_dir (str, default=None): Path to the non-salient dataset directory.
        --save_plot_path (str, default="plot.png"): Path to save the training plot.
        --model_save_path (str, default="model.pth"): Path to save the trained model.
        --model_load_path (str, default="model.pth"): Path to load a pre-trained model.
        --epochs (int, default=10): Number of training epochs.
        --learning_rate (float, default=1e-4): Learning rate for the optimizer.
        --batch_size (int, default=128): Batch size for training.
        --num_workers (int, default=0): Number of data loading workers.
    """
    parser = argparse.ArgumentParser(description="Train or evaluate a region classifier.")

    # General flags
    parser.add_argument(
        "--train_flag", action="store_true", help="Set this flag to enable training mode."
    )
    parser.add_argument(
        "--save_plot_flag", action="store_true", help="Set this flag to save training plots."
    )
    parser.add_argument(
        "--vram", action="store_true", help="Load entire dataset to VRAM for maximum speed."
    )

    # Paths
    # TODO: update the RC training pipeline to automatically handle file paths within /training_directory
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--vram_cache_path", type=str, default=None, help="Path to directory for caching VRAM tensors (e.g., on NVMe drive)."
    )
    
    parser.add_argument(
        "--broken_files_path", type=str, default=None, help="Path to the broken files list.",
    )
    
    parser.add_argument(
        "--non_salient_data_dir", type=str, default=None, help="Path to the non-salient dataset directory.",
    )
    parser.add_argument(
        "--save_plot_path", type=str, default="plot.png", help="Path to save the training plot."
    )
    parser.add_argument(
        "--model_save_path", type=str, default="model.pth", help="Path to save the trained model."
    )
    parser.add_argument(
        "--model_load_path", type=str,default="model.pth", help="Path to load the pre-trained model.",
    )

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loading workers (0 for HDD)."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.empty_cache()

    # Create the classifier object (VRAM or standard)
    ClassifierClass = TrainRegionClassifierFullVRAM if args.vram else TrainRegionClassifier
    
    # Prepare kwargs based on class type
    classifier_kwargs = {
        "data_path": args.data_dir,
        "broken_files_path": args.broken_files_path,
        "non_salient_data_path": args.non_salient_data_dir,
        "save_plot_flag": args.save_plot_flag,
        "save_plot_path": args.save_plot_path,
    }
    
    # Add batch_size and num_workers for VRAM mode
    if args.vram:
        classifier_kwargs["batch_size"] = args.batch_size
        classifier_kwargs["num_workers"] = args.num_workers
        classifier_kwargs["vram_cache_path"] = args.vram_cache_path
    
    classifier = ClassifierClass(**classifier_kwargs)

    if args.train_flag:
        # Set best model save path (best val F1 during training saved here, and loaded before final save)
        classifier.best_model_path = args.model_save_path.replace('.pth', '_best.pth')
        classifier.train(epochs=args.epochs, learning_rate=args.learning_rate)
        # Load the best model and save it as the primary output
        import os
        if os.path.exists(classifier.best_model_path):
            classifier.load_model(path=classifier.best_model_path)
            print(f"Loaded best model from {classifier.best_model_path}")
        classifier.save_model(path=args.model_save_path)
    else:
        # Load the model for evaluation
        classifier.load_model(path=args.model_load_path)
        print("Model loaded successfully.")

    # Evaluate the model
    classifier.evaluate()
