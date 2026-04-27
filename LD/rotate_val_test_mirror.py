"""
Mirror an existing LD_training directory into a sunac-owned location, with
val/test images and labels rotated by a random per-image angle. Train split
is symlinked back to the original (rotation augmentation happens via YOLO
RandomPerspective at training time).

Output layout for each region:
    {out_root}/{region}/LD_training/
        dataset.yaml                  (updated path)
        train/images, train/labels    (symlinks back to source)
        val/images/<stem>.png         (rotated PNG)
        val/labels/<stem>.txt         (rotated YOLO labels, axis-aligned bbox of
                                       rotated original corners, clipped to canvas)
        test/...                      (same as val)

Boxes whose rotated axis-aligned bounding box falls below a small min-size or
goes fully outside the canvas are dropped.
"""
import argparse
import math
import os
import random
import shutil
import sys
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm
import yaml


GRAY = (114, 114, 114)
MIN_WH_FRAC = 1e-4  # boxes smaller than this fraction of image dim → drop


def rotate_one(args):
    src_img, src_lbl, dst_img, dst_lbl, angle = args
    img = cv2.imread(src_img)
    if img is None:
        return f"FAIL load {src_img}"
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (W, H), borderValue=GRAY)
    cv2.imwrite(dst_img, rotated)

    # Read labels: each line is `cls cx cy w h` normalized in [0,1]
    new_lines = []
    if os.path.exists(src_lbl):
        with open(src_lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = parts[0]
                cx, cy, w, h = map(float, parts[1:])
                cx_px, cy_px = cx * W, cy * H
                w_px, h_px = w * W, h * H
                x1, y1 = cx_px - w_px / 2, cy_px - h_px / 2
                x2, y2 = cx_px + w_px / 2, cy_px + h_px / 2
                corners = np.array([
                    [x1, y1, 1.0],
                    [x2, y1, 1.0],
                    [x2, y2, 1.0],
                    [x1, y2, 1.0],
                ])
                rc = corners @ M.T  # (4, 2)
                nx1 = max(0.0, rc[:, 0].min())
                ny1 = max(0.0, rc[:, 1].min())
                nx2 = min(float(W), rc[:, 0].max())
                ny2 = min(float(H), rc[:, 1].max())
                if nx2 - nx1 < W * MIN_WH_FRAC or ny2 - ny1 < H * MIN_WH_FRAC:
                    continue
                ncx = (nx1 + nx2) / 2 / W
                ncy = (ny1 + ny2) / 2 / H
                nw = (nx2 - nx1) / W
                nh = (ny2 - ny1) / H
                new_lines.append(f"{cls} {ncx} {ncy} {nw} {nh}")
    with open(dst_lbl, "w") as f:
        if new_lines:
            f.write("\n".join(new_lines) + "\n")
    return None


def mirror_region(region, src_root, out_root, angle_lo, angle_hi, workers, seed, fixed_angles=None):
    src_ld = os.path.join(src_root, region, "LD_training")
    dst_ld = os.path.join(out_root, region, "LD_training")
    if not os.path.isdir(src_ld):
        print(f"[{region}] no src LD_training, skipping", flush=True)
        return
    os.makedirs(dst_ld, exist_ok=True)

    # train: symlink dirs back to source
    for sub in ("train",):
        src_sub = os.path.join(src_ld, sub)
        dst_sub = os.path.join(dst_ld, sub)
        if os.path.exists(dst_sub) or os.path.islink(dst_sub):
            if os.path.islink(dst_sub):
                os.unlink(dst_sub)
            elif os.path.isdir(dst_sub):
                shutil.rmtree(dst_sub)
        os.symlink(src_sub, dst_sub)

    # dataset.yaml: rewrite with new path
    src_yaml = os.path.join(src_ld, "dataset.yaml")
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = dst_ld
    with open(os.path.join(dst_ld, "dataset.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # val and test: rotate
    rng = random.Random(seed + hash(region) % (1 << 30))
    jobs = []
    for split in ("val", "test"):
        src_imgs_dir = os.path.join(src_ld, split, "images")
        src_lbls_dir = os.path.join(src_ld, split, "labels")
        dst_imgs_dir = os.path.join(dst_ld, split, "images")
        dst_lbls_dir = os.path.join(dst_ld, split, "labels")
        os.makedirs(dst_imgs_dir, exist_ok=True)
        os.makedirs(dst_lbls_dir, exist_ok=True)
        if not os.path.isdir(src_imgs_dir):
            continue
        for fname in sorted(os.listdir(src_imgs_dir)):
            if not fname.endswith(".png"):
                continue
            stem = fname[:-4]
            src_img = os.path.join(src_imgs_dir, fname)
            src_lbl = os.path.join(src_lbls_dir, stem + ".txt")
            if fixed_angles is None:
                dst_img = os.path.join(dst_imgs_dir, fname)
                dst_lbl = os.path.join(dst_lbls_dir, stem + ".txt")
                if os.path.exists(dst_img) and os.path.exists(dst_lbl):
                    continue
                angle = rng.uniform(angle_lo, angle_hi)
                jobs.append((src_img, src_lbl, dst_img, dst_lbl, angle))
            else:
                for ang in fixed_angles:
                    suffix = f"_a{int(round(ang)):+04d}"  # e.g. _a+045, _a-090
                    dst_img = os.path.join(dst_imgs_dir, stem + suffix + ".png")
                    dst_lbl = os.path.join(dst_lbls_dir, stem + suffix + ".txt")
                    if os.path.exists(dst_img) and os.path.exists(dst_lbl):
                        continue
                    jobs.append((src_img, src_lbl, dst_img, dst_lbl, float(ang)))

    print(f"[{region}] rotating {len(jobs)} val+test images", flush=True)
    if not jobs:
        return
    with Pool(workers) as pool:
        for err in tqdm(pool.imap_unordered(rotate_one, jobs), total=len(jobs), desc=region):
            if err:
                print(err)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--src_root", default="/mnt/sda2/training")
    p.add_argument("--out_root", default="/mnt/sda2/sunac_training_rotval")
    p.add_argument("--angle_lo", type=float, default=-180.0)
    p.add_argument("--angle_hi", type=float, default=180.0)
    p.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed_angles", type=float, nargs="+", default=None,
                   help="If provided, generate one rotated copy per angle for each val/test source image, "
                        "instead of one random angle per image. Output filenames get a _a±NNN suffix.")
    args = p.parse_args()

    for r in args.regions:
        mirror_region(r, args.src_root, args.out_root, args.angle_lo, args.angle_hi,
                      args.workers, args.seed, fixed_angles=args.fixed_angles)


if __name__ == "__main__":
    main()
