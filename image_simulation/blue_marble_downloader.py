import os
import requests
from itertools import product
import argparse

from requests.exceptions import ChunkedEncodingError
from tqdm import tqdm
from time import sleep

from image_simulation.blue_marble_simulator import MONTH_NAMES


BASE_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords"
DATASET_IDS_BY_MONTH = ["73938", "73967", "73992", "74017", "74042", "76487", "74092", "74117", "74142", "74167", "74192", "74218"]
RESOLUTION = "3x21600x21600"


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download the entire Blue Marble Next Generation dataset, at the highest resolution."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(__file__, "../blue_marble"),
        help="The directory to save the downloaded images to.",
    )
    return parser.parse_args()


def download_image(url: str, output_path: str, max_retries=100) -> None:
    """
    Download an image from a URL to a file.
    If the download fails, retry up to max_retries times.
    Each retry will resume the download from where it left off.

    :param url: The URL of the image to download.
    :param output_path: The path to save the downloaded image to.
    :param max_retries: The maximum number of times to retry the download.
    """
    # get file metadata
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
    }
    response = requests.head(url, headers=headers)
    response.raise_for_status()
    total_length = response.headers.get("content-length")
    assert total_length is not None
    assert response.headers.get("Content-Type") == "image/png"
    assert response.headers.get("Accept-Ranges") == "bytes"

    with open(output_path, "wb") as f, tqdm(total=int(total_length), unit="B", unit_scale=True,
                                            unit_divisor=1024, desc="Downloading") as pbar:
        for retry in range(max_retries):
            try:
                headers["Range"] = f"bytes={pbar.n}-"
                response = requests.get(url, headers=headers, stream=True, timeout=100)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                print(f"Download complete after {retry} retries")
                return
            except ChunkedEncodingError:
                sleep(5)
                continue

    print(f"Failed to download: {url}")
    os.remove(output_path)


def main():
    """
    Script entry point.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for i, (month_name, dataset_id) in enumerate(zip(MONTH_NAMES, DATASET_IDS_BY_MONTH)):
        month_dir = os.path.join(args.output_dir, month_name)
        os.makedirs(month_dir, exist_ok=True)

        dataset_section_id = dataset_id[:2] + "000"
        for letter, number in product("ABCD", range(1, 3)):
            image_section_id = f"{letter}{number}"
            url = f"{BASE_URL}/{dataset_section_id}/{dataset_id}/world.2004{(i + 1):02d}.{RESOLUTION}.{image_section_id}.png"
            output_path = os.path.join(month_dir, f"{letter}{number}.png")

            print(f"Downloading: {month_name} {image_section_id}")
            download_image(url, output_path)


if __name__ == "__main__":
    main()
