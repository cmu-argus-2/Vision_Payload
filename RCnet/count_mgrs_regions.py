"""
Counts the number of occurrences of each MGRS region in each image.

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000_lat_lon.npz
    - ...

This scipy will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000_mgrs_counts.json
    - ...
"""

import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
from itertools import starmap
from multiprocessing import Pool, cpu_count
from typing import ClassVar, Generator, Tuple

import numpy as np
from tqdm import tqdm

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: CuPy not available. Falling back to CPU computation.")
    print("Install CuPy for GPU acceleration: pip install cupy-cuda11x (replace 11x with your CUDA version)")

from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import calculate_mgrs_zones
from utils.function_utils import unpack_and_call

MGRS_COUNTS_SUFFIX = "_mgrs_counts.json"

@dataclass
class GeotaggedImage:
    """
    A class representing an image alongside lat/lon coordinates for each pixel.

    Attributes:
        image: A numpy array of shape CameraModel.RESOLUTION + (3,) containing the RGB image.
        lat_lon: A numpy array of shape CameraModel.RESOLUTION + (2,) containing the latitudes and longitudes for each
                 pixel, or np.nan if the pixel does not intersect the Earth.
    """

    IMAGE_SUFFIX: ClassVar[str] = ".png"
    LAT_LON_SUFFIX: ClassVar[str] = "_lat_lon.npz"

    image: np.ndarray
    lat_lon: np.ndarray

    def assert_invariants(self) -> None:
        """
        :raises AssertionError: If the image or lat/lon coordinates are invalid.
        """
        assert self.image is not None and self.lat_lon is not None
        # Allow flexible image dimensions (not just CameraModel.OUTPUT_SHAPE)
        assert len(self.image.shape) == 3 and self.image.shape[2] == 3, \
            f"Image must have shape (height, width, 3), got {self.image.shape}"
        assert self.image.dtype == CameraModel.DTYPE
        # Verify lat_lon matches image dimensions
        expected_lat_lon_shape = self.image.shape[:2] + (2,)
        assert self.lat_lon.shape == expected_lat_lon_shape, \
            f"lat_lon shape {self.lat_lon.shape} must match image shape {expected_lat_lon_shape}"
        assert np.issubdtype(self.lat_lon.dtype, np.floating)

    def save(self, region: str, file_prefix: str) -> None:
        """
        Save the image and lat/lon coordinates to the specified region and file prefix.

        :param region: The MGRS region.
        :param file_prefix: The prefix for the output files.
        """
        self.assert_invariants()

        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        region_dir = os.path.join(training_dir, region)
        os.makedirs(region_dir, exist_ok=True)

        cv2.imwrite(
            os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.IMAGE_SUFFIX}"),
            cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
        )
        np.savez_compressed(
            os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.LAT_LON_SUFFIX}"),
            lat_lon=self.lat_lon,
        )

    @staticmethod
    def load(region: str, file_prefix: str, delete_bad_files: bool = True) -> "GeotaggedImage":
        """
        Load the image and lat/lon coordinates from the specified region and file prefix.

        :param region: The MGRS region.
        :param file_prefix: The prefix for the output files.
        :param delete_bad_files: Whether to delete the files if they are malformed. This is very useful since it will
                                 take a long time to find which files are malformed later.
        :return: A GeotaggedImage object containing the loaded image and lat/lon coordinates.
        """
        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        region_dir = os.path.join(training_dir, region)

        img_path = os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.IMAGE_SUFFIX}")
        lat_lon_path = os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.LAT_LON_SUFFIX}")
        if not os.path.exists(img_path) or not os.path.exists(lat_lon_path):
            raise FileNotFoundError(
                f"Image or lat/lon file not found for region {region} with prefix {file_prefix}."
            )

        try:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            with np.load(lat_lon_path) as data:
                lat_lon = data["lat_lon"]

            geotagged_image = GeotaggedImage(image, lat_lon)
            geotagged_image.assert_invariants()
            return geotagged_image
        except Exception:
            if delete_bad_files:
                print(f"Warning: malformed image or lat/lon file for {region=}, {file_prefix=}. Deleting.")
                os.remove(img_path)
                os.remove(lat_lon_path)
            raise

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for counting MGRS regions.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Counts the number of occurrences of each MGRS region in each image."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to count occurrences for.",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip. This takes precedence over --regions.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file if it exists."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume counting MGRS regions for all requests that failed in the previous run.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.5 * cpu_count()),
        help="Number of processes to use for counting MGRS regions in parallel.",
    )

    return parser.parse_args()


def setup_region_dir(region_id: str, overwrite: bool, resume: bool) -> bool:
    """
    Set up the region directory for counting MGRS regions.

    :param region_id: The region ID to set up.
    :param overwrite: Whether to overwrite existing files. Cannot be True if resume is also True.
    :param resume: Whether to resume counting. Cannot be True if overwrite is also True.
    :return: True if the region directory is set up successfully, False otherwise.
    """
    assert not (resume and overwrite), "resume and overwrite cannot be used together."

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    print(f"DEBUG: USER_CONFIG_PATH={USER_CONFIG_PATH}")
    print(f"DEBUG: training_dir={training_dir}")
    region_dir = os.path.join(training_dir, region_id)
    print(f"DEBUG: region_dir={region_dir}")
    existing_files = [
        file_name for file_name in os.listdir(region_dir) if file_name.endswith(MGRS_COUNTS_SUFFIX)
    ]

    if len(existing_files) == 0:
        return True

    if overwrite:
        for file_name in existing_files:
            os.remove(os.path.join(region_dir, file_name))
        return True

    return True


def count_mgrs_regions(region_id: str, file_prefix: str) -> None:
    """
    Count the number of occurrences of each MGRS region in a geotagged image.

    :param region_id: The region ID to process.
    :param file_prefix: The file prefix for the geotagged image.
    """
    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    lat_lon_path = os.path.join(
        training_dir, region_id, f"{file_prefix}{GeotaggedImage.LAT_LON_SUFFIX}"
    )
    with np.load(lat_lon_path) as data:
        lat_lon = data["lat_lon"]

    mgrs_regions = calculate_mgrs_zones(lat_lon)
    present_regions = np.unique(mgrs_regions)
    counts = {
        str(region, encoding="ascii"): int(np.sum(mgrs_regions == region))
        for region in present_regions
    }

    output_path = os.path.join(training_dir, region_id, f"{file_prefix}{MGRS_COUNTS_SUFFIX}")
    with open(output_path, "w") as f:
        json.dump(counts, f, indent=4)


def main() -> None:
    """
    Script entry point.
    """
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("Cannot use --overwrite and --resume at the same time.")
    regions = sorted(set(args.regions) - set(args.skip_regions))
    args.num_processes = min(args.num_processes, len(regions))

    for region in tqdm(regions, desc="Setting up region directories"):
        if not setup_region_dir(region, args.overwrite, args.resume):
            print(
                f"Output files for {region} already exist. Set --overwrite to clear any existing data."
            )
            return

    def get_requests_generator() -> Generator[Tuple[str, str], None, None]:
        """
        :return: A generator that yields tuples of (region, file_prefix) for each request.
        """
        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        for region in regions:
            region_dir = os.path.join(training_dir, region)

            file_prefixes_generator = (
                file_name[: -len(GeotaggedImage.LAT_LON_SUFFIX)]
                for file_name in sorted(os.listdir(region_dir))
                if file_name.endswith(GeotaggedImage.LAT_LON_SUFFIX)
            )
            if args.resume:
                file_prefixes_generator = (
                    file_prefix
                    for file_prefix in file_prefixes_generator
                    if not os.path.exists(
                        os.path.join(region_dir, f"{file_prefix}{MGRS_COUNTS_SUFFIX}")
                    )
                )

            for file_prefix in file_prefixes_generator:
                yield region, file_prefix

    total_requests = sum(1 for _ in get_requests_generator())
    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        partial(unpack_and_call, count_mgrs_regions),
                        get_requests_generator(),
                        chunksize=1,
                    ),
                    total=total_requests,
                    desc="Counting MGRS regions",
                )
            )
    else:
        list(
            tqdm(
                starmap(count_mgrs_regions, get_requests_generator()),
                desc="Counting MGRS regions",
                total=total_requests,
            )
        )


if __name__ == "__main__":
    main()
