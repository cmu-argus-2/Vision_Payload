"""
Select K spatially-stratified landmarks from a region's full bounding-box catalog
via K-means on (longitude, latitude) centroids. For each cluster, pick the landmark
whose centroid is closest to the cluster centroid.

Input/output schema (6 cols, header preserved):
  Centroid Longitude, Centroid Latitude,
  Top-Left Longitude, Top-Left Latitude,
  Bottom-Right Longitude, Bottom-Right Latitude
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Source CSV with full catalog")
    p.add_argument("--dst", required=True, help="Destination CSV with selected K landmarks")
    p.add_argument("-k", type=int, default=100, help="Number of landmarks to select")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    with src.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[0].strip()]

    if len(rows) < args.k:
        sys.exit(f"Source has {len(rows)} rows, need at least {args.k}")

    coords = np.array([[float(r[0]), float(r[1])] for r in rows])  # (lon, lat)

    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    labels = km.fit_predict(coords)
    centers = km.cluster_centers_

    selected_idx = []
    for c in range(args.k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(coords[members] - centers[c], axis=1)
        selected_idx.append(int(members[dists.argmin()]))

    selected_idx.sort()
    selected = [rows[i] for i in selected_idx]

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(selected)

    sel = np.array([[float(r[0]), float(r[1])] for r in selected])
    print(f"Wrote {len(selected)} landmarks to {dst}")
    print(f"  source extent: lon[{coords[:,0].min():.2f},{coords[:,0].max():.2f}] "
          f"lat[{coords[:,1].min():.2f},{coords[:,1].max():.2f}]")
    print(f"  selected extent: lon[{sel[:,0].min():.2f},{sel[:,0].max():.2f}] "
          f"lat[{sel[:,1].min():.2f},{sel[:,1].max():.2f}]")
    print(f"  selected lon std: {sel[:,0].std():.2f} (vs source {coords[:,0].std():.2f})")
    print(f"  selected lat std: {sel[:,1].std():.2f} (vs source {coords[:,1].std():.2f})")


if __name__ == "__main__":
    main()
