"""
Main script for the RCnet training module.

This script initializes and runs the training pipeline
"""

import argparse

import torch

from VisionTrainingGround.RCnet.src.region_classifier import TrainRegionClassifier


def parse_args():
    """
    Parse command-line arguments for training or evaluating the image classifier.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Command-line Arguments:
        --train_flag (bool): Enable training mode if set.
        --save_plot_flag (bool): Save training plots if set.
        --data_dir (str, required): Path to the dataset directory.
        --non_salient_data_dir (str, default=None): Path to the non-salient dataset directory.
        --save_plot_path (str, default="plot.png"): Path to save the training plot.
        --model_save_path (str, default="model.pth"): Path to save the trained model.
        --model_load_path (str, default="model.pth"): Path to load a pre-trained model.
        --epochs (int, default=10): Number of training epochs.
        --learning_rate (float, default=1e-3): Learning rate for the optimizer.
    """
    parser = argparse.ArgumentParser(description="Train or evaluate a region classifier.")

    # General flags
    parser.add_argument(
        "--train_flag", action="store_true", help="Set this flag to enable training mode."
    )
    parser.add_argument(
        "--save_plot_flag", action="store_true", help="Set this flag to save training plots."
    )

    # Paths
    # TODO: update the RC training pipeline to automatically handle file paths within /training_directory
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--non_salient_data_dir",
        type=str,
        default=None,
        help="Path to the non-salient dataset directory.",
    )
    parser.add_argument(
        "--save_plot_path", type=str, default="plot.png", help="Path to save the training plot."
    )
    parser.add_argument(
        "--model_save_path", type=str, default="model.pth", help="Path to save the trained model."
    )
    parser.add_argument(
        "--model_load_path",
        type=str,
        default="model.pth",
        help="Path to load the pre-trained model.",
    )

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.empty_cache()

    # Create the classifier object
    classifier = TrainRegionClassifier(
        data_path=args.data_dir,
        non_salient_data_path=args.non_salient_data_dir,
        save_plot_flag=args.save_plot_flag,
        save_plot_path=args.save_plot_path,
    )

    if args.train_flag:
        # Train the model
        classifier.train(epochs=args.epochs, learning_rate=args.learning_rate)
        classifier.save_model(path=args.model_save_path)
    else:
        # Load the model for evaluation
        classifier.load_model(path=args.model_load_path)
        print("Model loaded successfully.")

    # Evaluate the model
    classifier.evaluate()
