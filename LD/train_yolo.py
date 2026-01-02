"""
Train landmark detection YOLO models for the specified MGRS regions.

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - /LD_training
      - dataset.yaml
      - /train
        - /images
          - 00000.png (symlink)
          - ...
        - /labels
          - 00000.txt
          - ...
      - /test
        - ...
      - /val
        - ...

This scipy will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - yolo_model_weights.pt
    - yolo_training_results
      - ...
"""

import argparse
import os
import shutil

import numpy as np
from time import time
import torch
from tqdm import tqdm
from ultralytics import YOLO

from utils.config_utils import USER_CONFIG_PATH, load_config
from vision_inference.landmark_detector import LandmarkDetector
from VisionTrainingGround.LD.prepare_yolo_data import LD_TRAINING_DIR_NAME, YOLO_CONFIG_FILE_NAME


TRAINING_LOG_DIR_PREFIX = "yolo_training_results"


def parse_args():
    """
    Parse command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train landmark detection YOLO models for the specified MGRS regions."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to train landmark detection YOLO models for.",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip. This takes precedence over --regions.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--version", type=str, required=False, default="yolov8s", help="The YOLO version to use."
    )
    parser.add_argument(
        "--epochs", type=int, required=False, default=100, help="The number of training epochs."
    )
    return parser.parse_args()


def train_yolo(
    region: str,
    overwrite: bool,
    version: str,
    epochs: int,
) -> None:
    """
    Main function to initialize and train a YOLO model using specified command-line arguments.

    This function:
    - Determines the computing device (CPU or GPU).
    - Loads the YOLO model based on the version specified.
    - Sets up the training configuration and runs the training process.
    - Saves the trained model.

    Arguments:
    - region: The MGRS region to train the model for.
    - overwrite: Whether to overwrite the output files if they exist.
    - version: The YOLO model version to use.
    - epochs: The number of epochs for training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device=} for training {region=}")

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    region_dir = os.path.join(training_dir, region)
    training_log_dir = os.path.join(region_dir, TRAINING_LOG_DIR_PREFIX)
    output_file = os.path.join(
        training_dir, LandmarkDetector.get_LD_model_weights_relative_path(region)
    )

    if os.path.exists(output_file):
        if not overwrite:
            raise FileExistsError(f"Output file {output_file} already exists.")

        os.remove(output_file)

    model = YOLO(f"{version}.pt")
    yolo_config_path = os.path.join(
        region_dir, LD_TRAINING_DIR_NAME, YOLO_CONFIG_FILE_NAME
    )
    # pylint: disable=unused-variable
    results = model.train(
        data=yolo_config_path,
        # The result files are saved in os.path.join(__file__, "../", project, name)
        project=f"{TRAINING_LOG_DIR_PREFIX}_{region}",
        name=f"{TRAINING_LOG_DIR_PREFIX}_{region}_{time()}",
        # Image augmentation parameters
        degrees=0,
        scale=0,
        fliplr=0,
        mosaic=0,
        perspective=0,
        # Training parameters
        # Ultralytics YOLO only supports training on square images, so it will pad the image to (4608, 4608) with gray
        imgsz=4600,  # Single integer (longest dimension)
        rect=True, 
        batch=2,
        plots=True,
        save=True,
        resume=False,
        epochs=epochs,
        device=device,
    )

    # Move the logs from the directory that they are saved into by YOLO, to where we actually want them
    # TODO: figure out why this isn't working
    # output_log_dir = os.path.join(__file__, "../", f"{TRAINING_LOG_DIR_PREFIX}_{region}")
    # for directory in os.listdir(output_log_dir):
    #     shutil.move(
    #         os.path.join(output_log_dir, directory),
    #         os.path.join(training_log_dir, directory),
    #     )
    # shutil.rmtree(output_log_dir)


def main() -> None:
    """
    Script entry point.
    """
    args = parse_args()
    regions = sorted(set(args.regions) - set(args.skip_regions))

    for region in tqdm(regions, desc="Training YOLO models"):
        try:
            train_yolo(region, args.overwrite, args.version, args.epochs)
        except Exception as e:
            print(f"Error training YOLO model for {region}: {e}")
            continue


if __name__ == "__main__":
    main()
