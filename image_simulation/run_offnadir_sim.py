"""
Simulate off-NADIR views of validation/test images with optional color tints.

For each image in the chosen split, the script:
1. Reads its _lat_lon.npz to get the center lat/lon.
2. Places the satellite at --altitude_km directly above that center.
3. For each (off-NADIR angle, tint) combination, renders an image via the
   EarthImageSimulator (which queries the GeoTIFF cache at the configured
   geotiff_folder) and applies the chosen tint.

Output layout:
    <output_dir>/
        <region>/
            <image_stem>/
                roll{ANGLE}_{TINT}.png      # e.g. roll0_orange.png
"""
import argparse
import os
import random
import sys
from multiprocessing import cpu_count, Pool
from functools import partial

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_simulation.camera_model import CameraModelManager
from image_simulation.color_tint import apply_tint, TINT_NAMES
from image_simulation.png_simulator import SourceImage, simulate_from_source
from utils.config_utils import load_config
from utils.earth_utils import lat_lon_to_ecef, get_nadir_rotation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate off-NADIR + color-tinted views of val/test images.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Training directory containing {region}/{stem}.png and {stem}_lat_lon.npz.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write simulated images.")
    p.add_argument("--split", choices=["val", "test"], required=True,
                   help="Which held-out split to use (same 42 seed as training).")
    p.add_argument("--num_images", type=int, default=None,
                   help="Limit simulation to the first N images of the split. Default: all.")
    p.add_argument("--altitude_km", type=float, default=590.0,
                   help="Satellite altitude above the image center, km. Default: 590.")
    p.add_argument("--off_nadir_degs", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0],
                   help="Roll angles (deg) to simulate (fixed mode). Default: 0 5 10 15.")
    p.add_argument("--angle_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="Random mode: each image gets one angle sampled uniformly in [MIN, MAX] deg. "
                        "Overrides --off_nadir_degs.")
    p.add_argument("--tints", type=str, nargs="+", default=TINT_NAMES,
                   help=f"Tints to apply. Default: {TINT_NAMES}")
    p.add_argument("--camera", type=str, default="x+",
                   help="Camera name (x+, y+, x-, y-). Default: x+.")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_processes", type=int, default=max(1, int(0.5 * cpu_count())),
                   help="Parallel worker count. Default: 50%% of cores.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip output files that already exist.")
    return p.parse_args()


def collect_split_files(data_dir: str, regions: list, split: str,
                        train_ratio: float, val_ratio: float, seed: int) -> list:
    """Replicate the same deterministic split used by the RC training dataloader."""
    all_files = []
    salient = set(regions)
    for folder in os.listdir(data_dir):
        if folder not in salient:
            continue
        rdir = os.path.join(data_dir, folder)
        if not os.path.isdir(rdir):
            continue
        for fname in sorted(os.listdir(rdir)):
            if not (fname.endswith(".png") or fname.endswith(".jpg")):
                continue
            if not fname[3].isdigit():
                continue
            stem = fname.rsplit(".", 1)[0]
            ll_path = os.path.join(rdir, f"{stem}_lat_lon.npz")
            json_path = os.path.join(rdir, f"{stem}_mgrs_counts.json")
            if os.path.exists(ll_path) and os.path.exists(json_path):
                all_files.append((folder, os.path.join(rdir, fname), ll_path, stem))
    random.seed(seed)
    random.shuffle(all_files)
    n = len(all_files)
    tr = int(train_ratio * n)
    vl = int(val_ratio * n)
    if split == "val":
        return all_files[tr:tr + vl]
    return all_files[tr + vl:]  # test


def compute_satellite_state(center_lat_lon: np.ndarray, altitude_m: float) -> np.ndarray:
    """Return 6-dim ECEF state [pos, vel] for a circular-orbit satellite directly over center."""
    surface_ecef = lat_lon_to_ecef(center_lat_lon)                  # shape (3,)
    r_surface = np.linalg.norm(surface_ecef)
    position_ecef = surface_ecef * (r_surface + altitude_m) / r_surface

    # Circular orbit velocity magnitude
    MU = 3.986004418e14
    v_mag = np.sqrt(MU / np.linalg.norm(position_ecef))

    # Velocity direction: perpendicular to position, in the equatorial/easterly sense
    # Pick a tangent direction (cross with Earth z-axis, then normalize)
    z_axis = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(z_axis, position_ecef)
    tangent /= np.linalg.norm(tangent)
    velocity_ecef = tangent * v_mag
    return np.concatenate([position_ecef, velocity_ecef])


def simulate_one(
    file_entry: tuple,
    output_dir: str,
    altitude_m: float,
    off_nadir_degs: list,
    angle_range: tuple,
    tints: list,
    camera_name: str,
    skip_existing: bool,
    sim_seed: int,
) -> int:
    """Simulate all (angle, tint) variants for one source image. Returns rendered count."""
    region, img_path, ll_path, stem = file_entry

    # In random-angle mode, sample one angle per source image (deterministic per stem via seed).
    if angle_range is not None:
        rng = np.random.default_rng(sim_seed + abs(hash(stem)) % (2**32))
        angles_for_this_image = [float(rng.uniform(angle_range[0], angle_range[1]))]
    else:
        angles_for_this_image = list(off_nadir_degs)

    # Lazy-init the camera manager once per process
    global _CAMERAS
    try:
        _CAMERAS
    except NameError:
        _CAMERAS = CameraModelManager()
    camera = _CAMERAS[camera_name]

    # Load the source PNG + lat/lon grid for this image
    try:
        source = SourceImage.load(img_path, ll_path)
    except Exception as e:
        print(f"[WARN] load failed for {region}/{stem}: {e}")
        return 0

    # Center lat/lon from the lat/lon grid
    H_src, W_src = source.lat_lon.shape[:2]
    center_lat_lon = np.asarray(source.lat_lon[H_src // 2, W_src // 2], dtype=np.float64)

    state = compute_satellite_state(center_lat_lon, altitude_m)
    position_ecef = state[:3]
    ecef_R_body_nadir = get_nadir_rotation(state, nadir_axis="x+")

    out_region_dir = os.path.join(output_dir, region, stem)
    os.makedirs(out_region_dir, exist_ok=True)

    rendered = 0
    for angle in angles_for_this_image:
        # Pure roll about body-y; rotation applied to camera so the optical axis tilts
        roll = Rotation.from_euler("y", float(angle), degrees=True).as_matrix()
        ecef_R_body = ecef_R_body_nadir @ roll

        # Filename keeps full angle to 1 decimal so random angles stay distinct
        angle_tag = f"roll{angle:.1f}"

        # Check outputs first for skip_existing
        per_angle_outs = [
            os.path.join(out_region_dir, f"{angle_tag}_{t}.png") for t in tints
        ]
        if skip_existing and all(os.path.exists(p) for p in per_angle_outs):
            continue

        try:
            rendered_img = simulate_from_source(source, position_ecef, ecef_R_body, camera)
        except Exception as e:
            print(f"[WARN] sim failed for {region}/{stem} angle={angle}: {e}")
            continue

        for tint, out_path in zip(tints, per_angle_outs):
            if skip_existing and os.path.exists(out_path):
                continue
            tinted = apply_tint(rendered_img, tint)
            cv2.imwrite(out_path, cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR))
            rendered += 1
    return rendered


def main() -> None:
    args = parse_args()
    regions = load_config()["vision"]["salient_mgrs_region_ids"]
    files = collect_split_files(args.data_dir, regions, args.split,
                                args.train_ratio, args.val_ratio, args.seed)
    if args.num_images is not None:
        files = files[:args.num_images]
    print(f"[Sim] {args.split} split: {len(files)} source images")
    print(f"[Sim] angles={args.off_nadir_degs}, tints={args.tints}, camera={args.camera}, alt={args.altitude_km} km")

    altitude_m = args.altitude_km * 1000.0
    angle_range = tuple(args.angle_range) if args.angle_range is not None else None
    if angle_range is not None:
        print(f"[Sim] random-angle mode: uniform in [{angle_range[0]}, {angle_range[1]}] deg per image")
    work = partial(simulate_one,
                   output_dir=args.output_dir,
                   altitude_m=altitude_m,
                   off_nadir_degs=args.off_nadir_degs,
                   angle_range=angle_range,
                   tints=args.tints,
                   camera_name=args.camera,
                   skip_existing=args.skip_existing,
                   sim_seed=args.seed)

    if args.num_processes <= 1:
        total = 0
        for f in tqdm(files, desc="simulating"):
            total += work(f)
    else:
        with Pool(args.num_processes) as pool:
            total = sum(tqdm(pool.imap_unordered(work, files, chunksize=1),
                              total=len(files), desc="simulating"))
    print(f"[Sim] Done. Rendered {total} images to {args.output_dir}")


if __name__ == "__main__":
    main()
