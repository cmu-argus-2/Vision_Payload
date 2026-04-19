"""
Lightweight off-NADIR Earth image simulator that sources color data from existing
PNG images paired with per-pixel lat/lon grids (the _lat_lon.npz files produced
during reprojection), rather than querying GeoTIFFs directly.

This avoids the need to maintain a reprojected GeoTIFF library while still
supporting arbitrary off-NADIR camera attitudes within the footprint of the
source image.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from image_simulation.camera_model import CameraModel
from utils.earth_utils import ecef_to_lat_lon, intersect_ellipsoid


@dataclass
class SourceImage:
    """A reprojected source image with its per-pixel lat/lon grid."""
    image: np.ndarray       # (H, W, 3) uint8, RGB
    lat_lon: np.ndarray     # (H, W, 2) float, [:,:,0]=lat, [:,:,1]=lon

    @staticmethod
    def load(png_path: str, npz_path: str) -> "SourceImage":
        from PIL import Image
        img = np.asarray(Image.open(png_path).convert("RGB"))  # (H, W, 3)
        ll = np.load(npz_path)["lat_lon"]                       # (H, W, 2)
        assert img.shape[:2] == ll.shape[:2], (
            f"image/latlon shape mismatch: {img.shape[:2]} vs {ll.shape[:2]}"
        )
        return SourceImage(image=img, lat_lon=ll)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """(lat_min, lat_max, lon_min, lon_max) covering the source image."""
        lat = self.lat_lon[..., 0]
        lon = self.lat_lon[..., 1]
        return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def _latlon_to_source_uv(
    lat_lon_query: np.ndarray,
    source: SourceImage,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (lat, lon) query points to fractional (u, v) pixel coords inside the
    source image, assuming the source lat/lon grid is linearly spaced (true after
    EPSG:4326 reprojection).

    :param lat_lon_query: shape (..., 2), float
    :param source: SourceImage
    :return: (us, vs, valid_mask) — floats with source.lat_lon.shape[:-1], plus bool mask.
    """
    H, W = source.lat_lon.shape[:2]
    # Reprojected to EPSG:4326: the grid is regular in (lat, lon).
    # Infer spacing from the corner pixels.
    lat_top = source.lat_lon[0, W // 2, 0]           # near top row
    lat_bot = source.lat_lon[H - 1, W // 2, 0]       # near bottom row
    lon_lft = source.lat_lon[H // 2, 0, 1]
    lon_rgt = source.lat_lon[H // 2, W - 1, 1]

    lat_q = lat_lon_query[..., 0]
    lon_q = lat_lon_query[..., 1]

    # v = row index (0 = top). Lat usually decreases with row.
    vs = (lat_q - lat_top) / (lat_bot - lat_top) * (H - 1)
    us = (lon_q - lon_lft) / (lon_rgt - lon_lft) * (W - 1)

    valid = (
        np.isfinite(lat_q) & np.isfinite(lon_q)
        & (us >= 0) & (us <= W - 1)
        & (vs >= 0) & (vs <= H - 1)
    )
    return us, vs, valid


def _bilinear_sample(image: np.ndarray, us: np.ndarray, vs: np.ndarray,
                     valid: np.ndarray) -> np.ndarray:
    """Sample RGB bilinearly; invalid pixels → 0."""
    H, W, C = image.shape
    out = np.zeros(us.shape + (C,), dtype=np.uint8)
    if not np.any(valid):
        return out
    us_valid = us[valid]
    vs_valid = vs[valid]
    # map_coordinates wants (y, x) ordered coords (= v, u)
    coords = np.stack([vs_valid, us_valid], axis=0)
    for c in range(C):
        vals = map_coordinates(image[..., c].astype(np.float32), coords,
                               order=1, mode="constant", cval=0.0)
        channel = np.zeros(us.shape, dtype=np.uint8)
        channel[valid] = np.clip(vals, 0, 255).astype(np.uint8)
        out[..., c] = channel
    return out


def simulate_from_source(
    source: SourceImage,
    position_ecef: np.ndarray,
    ecef_R_body: np.ndarray,
    camera: CameraModel,
) -> np.ndarray:
    """
    Render an image as seen by the given camera pose, sampling from the source
    PNG via the source's per-pixel lat/lon grid.

    :param source: SourceImage (loaded PNG + lat/lon grid).
    :param position_ecef: (3,) ECEF satellite position, meters.
    :param ecef_R_body: (3, 3) rotation matrix body → ECEF.
    :param camera: CameraModel (provides ray_directions_body + camera pose in body).
    :return: uint8 (H_cam, W_cam, 3) simulated image.
    """
    # Rays in camera frame → body → ECEF
    ray_dirs_body = camera.ray_directions_body()                     # (H, W, 3)
    ray_dirs_ecef = ray_dirs_body @ ecef_R_body.T                    # (H, W, 3)
    camera_pos_ecef = camera.get_camera_position(position_ecef, ecef_R_body)

    intersections = intersect_ellipsoid(ray_dirs_ecef, camera_pos_ecef)  # (H, W, 3)
    lat_lon = ecef_to_lat_lon(intersections)                              # (H, W, 2)

    us, vs, valid = _latlon_to_source_uv(lat_lon, source)
    return _bilinear_sample(source.image, us, vs, valid)
