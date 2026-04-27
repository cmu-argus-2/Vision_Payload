"""
Convert bonus GeoTIFFs to (PNG + _lat_lon.npz) pairs in the same on-disk format
the LD training prep expects, renumbered to start at offset 500 so they don't
collide with the existing 33S images.
"""
import argparse
import os
import sys
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm


def convert_one(args):
    src_tif, out_png, out_npz = args
    with rasterio.open(src_tif) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))
        H, W = rgb.shape[:2]
        transform = src.transform
        crs = src.crs

    cv2.imwrite(out_png, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Per-pixel UTM coords from the affine transform, then project to lat/lon.
    cols = np.arange(W)
    rows = np.arange(H)
    cc, rr = np.meshgrid(cols, rows)
    xs = transform.a * cc + transform.b * rr + transform.c
    ys = transform.d * cc + transform.e * rr + transform.f

    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(xs.ravel(), ys.ravel())
    lat_lon = np.stack([lat.reshape(H, W), lon.reshape(H, W)], axis=-1).astype(np.float64)

    np.savez(out_npz, lat_lon=lat_lon)
    return out_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default="/mnt/sda2/bonus_images/geotiffs/33S")
    p.add_argument("--out_dir", default="/mnt/sda2/sunac_training/33S")
    p.add_argument("--region", default="33S")
    p.add_argument("--src_prefix", default="l8_33S_")
    p.add_argument("--index_offset", type=int, default=500)
    p.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    src_files = sorted(
        f for f in os.listdir(args.src_dir)
        if f.startswith(args.src_prefix) and f.endswith(".tif")
    )

    jobs = []
    for f in src_files:
        stem = f[len(args.src_prefix):-len(".tif")]
        try:
            src_idx = int(stem)
        except ValueError:
            continue
        out_idx = src_idx + args.index_offset
        out_stem = f"l8_{args.region}_{out_idx:05d}"
        out_png = os.path.join(args.out_dir, out_stem + ".png")
        out_npz = os.path.join(args.out_dir, out_stem + "_lat_lon.npz")
        if os.path.exists(out_png) and os.path.exists(out_npz):
            continue
        jobs.append((os.path.join(args.src_dir, f), out_png, out_npz))

    print(f"Converting {len(jobs)} GeoTIFFs (skipping {len(src_files) - len(jobs)} already done) "
          f"with {args.workers} workers")

    if not jobs:
        return

    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(convert_one, jobs), total=len(jobs)):
            pass


if __name__ == "__main__":
    main()
