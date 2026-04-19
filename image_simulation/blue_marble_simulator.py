"""
Code for simulating the colors of the Earth's surface as seen from space using the Blue Marble Next Generation dataset.

The dataset is divided into directories by month, and each month directory contains an equirectangular projection
of the Earth's surface divided into 8 tiles arranged in a 2x4 grid as follows:
A1 B1 C1 D1
A2 B2 C2 D2
"""

import os
from typing import Tuple
from PIL import Image
from affine import Affine

import numpy as np

from utils.config_utils import USER_CONFIG_PATH, load_config

# disable the PIL image size limit
Image.MAX_IMAGE_PIXELS = None

MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
IMG_LAT_BOUNDS = {
    "1": (0, 90),
    "2": (-90, 0),
}
IMG_LON_BOUNDS = {
    "A": (-180, -90),
    "B": (-90, 0),
    "C": (0, 90),
    "D": (90, 180),
}


def get_blue_marble_img(
        month: str,
        img_name: str,
        query_bounds: Tuple[float, float, float, float] | None = None
) -> Tuple[np.ndarray, Affine]:
    """
    Get the portion of the Blue Marble image for the given month, image name, and the specified bounds.

    Args:
        month: The month of the Blue Marble Next Generation dataset to load from.
        img_name: The name of the image tile to load. Must be one of ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"].
        query_bounds: The bounds of the query in the form (min_lat, max_lat, min_lon, max_lon). If None, the entire
                      image will be returned. If the provided bounds extend beyond the bounds of the specified image,
                      the returned image will nonetheless be cropped to the image bounds.

    Returns:
        A tuple containing:
            - The requested portion of the Blue Marble image, as a numpy array.
            - An affine transformation matrix mapping lat/lon coordinates to pixel coordinates in the returned image.
    """
    path = os.path.join(load_config(USER_CONFIG_PATH)["blue_marble_directory"], month, f"{img_name}.png")
    assert os.path.exists(path), f"Blue Marble image not found: {path}"

    img_min_lat, img_max_lat = IMG_LAT_BOUNDS[img_name[1]]
    img_min_lon, img_max_lon = IMG_LON_BOUNDS[img_name[0]]

    # Limit the query bounds to the bounds of the image
    if query_bounds is not None:
        query_min_lat, query_max_lat, query_min_lon, query_max_lon = query_bounds
        min_lat = max(query_min_lat, img_min_lat)
        max_lat = min(query_max_lat, img_max_lat)
        min_lon = max(query_min_lon, img_min_lon)
        max_lon = min(query_max_lon, img_max_lon)
    else:
        min_lat, max_lat, min_lon, max_lon = img_min_lat, img_max_lat, img_min_lon, img_max_lon

    # use PIL to allow loading just the necessary portion of the image
    with Image.open(path) as img:
        width, height = img.size

        min_u = width * (min_lon - img_min_lon) / (img_max_lon - img_min_lon)
        max_u = width * (max_lon - img_min_lon) / (img_max_lon - img_min_lon)
        # Note that min_v corresponds to max_lat and max_v corresponds to min_lat
        # Also since the vertical axis is flipped, the coordinates are measured from img_max_lat instead of img_min_lat
        min_v = height * (img_max_lat - max_lat) / (img_max_lat - img_min_lat)
        max_v = height * (img_max_lat - min_lat) / (img_max_lat - img_min_lat)

        # round to integer pixel coordinates
        min_u = int(np.floor(min_u))
        max_u = int(np.ceil(max_u))
        min_v = int(np.floor(min_v))
        max_v = int(np.ceil(max_v))

        roi = np.array(img.crop((min_u, min_v, max_u, max_v)))

    # recompute bounds based on rounded pixel coordinates
    min_lat = img_max_lat - (max_v / height) * (img_max_lat - img_min_lat)
    max_lat = img_max_lat - (min_v / height) * (img_max_lat - img_min_lat)
    min_lon = img_min_lon + (min_u / width) * (img_max_lon - img_min_lon)
    max_lon = img_min_lon + (max_u / width) * (img_max_lon - img_min_lon)

    scale_u = (max_u - min_u) / (max_lon - min_lon)
    scale_v = (max_v - min_v) / (max_lat - min_lat)

    # maps (lat, lon) to (u, v) (i.e. width, height)
    transform = Affine(
        # u = a * lat + b * lon + c, lon = min_lon -> u = 0, lon = max_lon -> u = max_u - min_u
        a=0,
        b=scale_u,
        c=-scale_u * min_lon,
        # v = d * lat + e * lon + f, lat = min_lat -> v = max_v - min_v, lat = max_lat -> v = 0
        d=-scale_v,
        e=0,
        f=scale_v * max_lat,
    )

    return roi, transform


def query_blue_marble_pixel_colors(lat_lon: np.ndarray, month: str | None = None) -> np.ndarray:
    """
    Query the colors of the pixels at the given latitudes and longitudes.

    :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes of the pixels to query.
    :param month: The month to simulate. If None, a random month will be chosen.
    :return: A numpy array of shape (..., 3) containing the RGB values of the pixels.
    """
    assert lat_lon.shape[-1] == 2, "The last dimension of lat_lon must be 2."
    if lat_lon.ndim == 1:
        # special case for single query
        return query_blue_marble_pixel_colors(lat_lon[np.newaxis, :], month)[0, :]

    if month is None:
        month = np.random.choice(MONTH_NAMES)

    shape_prefix = lat_lon.shape[:-1]
    lat_lon = lat_lon.reshape(-1, 2)

    img_letters = np.full(lat_lon.shape[0], "", dtype=str)
    for letter in "ABCD":
        img_min_lon, img_max_lon = IMG_LON_BOUNDS[letter]
        img_letters[(img_min_lon <= lat_lon[:, 1]) & (lat_lon[:, 1] < img_max_lon)] = letter
    assert np.all(img_letters != ""), "Longitude out of bounds."

    img_numbers = np.where(lat_lon[:, 0] >= 0, "1", "2")
    img_names = np.char.add(img_letters, img_numbers)

    pixel_colors = np.zeros((lat_lon.shape[0], 3), dtype=np.uint8)
    for img_name in set(img_names):
        img_lat_lon = lat_lon[img_names == img_name, :]
        img_query_min_lat, img_query_min_lon = np.min(img_lat_lon, axis=0)
        img_query_max_lat, img_query_max_lon = np.max(img_lat_lon, axis=0)

        img_min_lat, img_max_lat = IMG_LAT_BOUNDS[img_name[1]]
        img_min_lon, img_max_lon = IMG_LON_BOUNDS[img_name[0]]
        assert img_query_min_lat >= img_min_lat and img_query_max_lat <= img_max_lat, "Latitude out of bounds."
        assert img_query_min_lon >= img_min_lon and img_query_max_lon <= img_max_lon, "Longitude out of bounds."

        img, transform = get_blue_marble_img(
            month,
            img_name,
            (img_query_min_lat, img_query_max_lat, img_query_min_lon, img_query_max_lon),
        )
        assert img.dtype == pixel_colors.dtype, "Image dtype does not match pixel_colors dtype."
        height, width = img.shape[:2]

        us, vs = transform * tuple(img_lat_lon.T)
        us = np.rint(us).astype(int)
        vs = np.rint(vs).astype(int)
        us[us == width] = width - 1
        vs[vs == height] = height - 1
        assert np.all((us >= 0) & (us < width)), "Pixel u-coordinate out of bounds."
        assert np.all((vs >= 0) & (vs < height)), "Pixel v-coordinate out of bounds."

        pixel_colors[img_names == img_name, :] = img[vs, us, :]

    return pixel_colors.reshape(shape_prefix + (3,))
