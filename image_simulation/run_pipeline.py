"""
Full pipeline runner: simulated image -> RC (region classifier) -> LD (YOLO landmark detector).

Walks a directory of simulated images produced by run_offnadir_sim.py, runs the RC
network to predict which MGRS regions are visible, then for each predicted region
runs the region-specific YOLO model to detect landmarks.

Outputs a JSON summary file with per-image RC and LD results plus aggregate stats
(RC recall vs source ground-truth region, LD landmark counts per region).

Typical usage:
    python image_simulation/run_pipeline.py \
      --sim_dir image_simulation/sample_outputs_random \
      --rc_model RCnet/output/model_a090_strict_best.pth \
      --ld_models_base /mnt/sda2/training \
      --output_json image_simulation/pipeline_results.json
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision_inference.region_classifier import (
    ClassifierEfficient,
    RegionClassifier,
)


# Output file name pattern: roll<angle>_<tint>.png
_SIM_NAME_RE = re.compile(r"^roll([0-9]+(?:\.[0-9]+)?)_([a-z]+)\.png$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sim_dir", required=True,
                   help="Directory of simulated images (output of run_offnadir_sim.py).")
    p.add_argument("--rc_model", required=True,
                   help="Path to trained RC model .pth (e.g. model_a090_strict_best.pth).")
    p.add_argument("--ld_models_base", default="/mnt/sda2/training",
                   help="Directory with {region}/yolo_model_weights.pt. Default: /mnt/sda2/training.")
    p.add_argument("--output_json", required=True,
                   help="Where to write the aggregated pipeline results.")
    p.add_argument("--rc_pred_thresh", type=float, default=0.5,
                   help="RC prediction threshold (probability > this = region present).")
    p.add_argument("--ld_conf_thresh", type=float, default=0.25,
                   help="YOLO confidence threshold for detections.")
    p.add_argument("--ld_imgsz", type=int, default=4608,
                   help="YOLO inference image size. Default 4608 (matches training).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap processed images for smoke tests.")
    p.add_argument("--source_data_dir", default="/mnt/sda2/training",
                   help="Dir containing original (NADIR, untinted) source PNGs per region. "
                        "Used to compute per-source LD baseline detections for comparison.")
    p.add_argument("--gt_labels_subdir", default="LD_training/val/labels",
                   help="Subpath under source_data_dir/{region} where YOLO GT .txt labels live. "
                        "If found, Level-2 metrics (sim vs. ground-truth class IDs) are included.")
    return p.parse_args()


def walk_sim_outputs(sim_dir: str):
    """Yield (source_region, stem, angle, tint, png_path) for each simulated image."""
    for region in sorted(os.listdir(sim_dir)):
        rdir = os.path.join(sim_dir, region)
        if not os.path.isdir(rdir):
            continue
        for stem in sorted(os.listdir(rdir)):
            sdir = os.path.join(rdir, stem)
            if not os.path.isdir(sdir):
                continue
            for fname in sorted(os.listdir(sdir)):
                m = _SIM_NAME_RE.match(fname)
                if not m:
                    continue
                yield region, stem, float(m.group(1)), m.group(2), os.path.join(sdir, fname)


def load_rc_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierEfficient().to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    tfm = transforms.Compose([
        transforms.Resize(RegionClassifier.DOWNSAMPLED_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=RegionClassifier.IMAGE_NET_MEAN, std=RegionClassifier.IMAGE_NET_STD),
    ])
    region_ids = sorted(RegionClassifier.load_region_ids())
    return model, tfm, region_ids, device


def rc_infer(model, tfm, region_ids, device, png_path: str, pred_thresh: float):
    img = Image.open(png_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).cpu().squeeze(0).numpy()
    predicted = [region_ids[i] for i, p in enumerate(probs) if p > pred_thresh]
    return predicted, probs.tolist()


# Cache YOLO models across calls
_LD_CACHE = {}
_LD_MISSING = set()


def load_ld_model(region: str, ld_models_base: str):
    if region in _LD_CACHE:
        return _LD_CACHE[region]
    if region in _LD_MISSING:
        return None
    path = os.path.join(ld_models_base, region, "yolo_model_weights.pt")
    if not os.path.exists(path):
        print(f"[Pipeline] no LD model for {region} at {path}, skipping.")
        _LD_MISSING.add(region)
        return None
    from ultralytics import YOLO
    model = YOLO(path)
    _LD_CACHE[region] = model
    return model


def ld_infer(model, png_path: str, conf_thresh: float, imgsz: int):
    results = model.predict(source=png_path, conf=conf_thresh, imgsz=imgsz, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy().tolist()
    conf = r.boxes.conf.cpu().numpy().tolist()
    cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
    return [{"class": c, "conf": float(cf), "bbox": b} for b, cf, c in zip(xyxy, conf, cls)]


def main() -> None:
    args = parse_args()
    print(f"[Pipeline] loading RC model {args.rc_model}")
    rc_model, rc_tfm, region_ids, device = load_rc_model(args.rc_model)

    entries = list(walk_sim_outputs(args.sim_dir))
    if args.limit:
        entries = entries[:args.limit]
    print(f"[Pipeline] processing {len(entries)} simulated images")

    per_image = []
    rc_hit = 0
    rc_total = 0
    ld_totals = defaultdict(int)
    baseline_cache = {}   # (source_region, stem) -> {"classes": set, "num": int}
    gt_cache = {}         # (source_region, stem) -> set of GT class ids

    for i, (source_region, stem, angle, tint, png_path) in enumerate(entries):
        predicted_regions, probs = rc_infer(rc_model, rc_tfm, region_ids, device,
                                            png_path, args.rc_pred_thresh)
        rc_correct = source_region in predicted_regions
        rc_hit += int(rc_correct)
        rc_total += 1

        # Compute LD baseline on the ORIGINAL source PNG for this (source_region, stem).
        # Only the source_region's own LD model is used for baseline (what "should" be there).
        baseline_key = (source_region, stem)
        if baseline_key not in baseline_cache:
            original_png = os.path.join(args.source_data_dir, source_region, f"{stem}.png")
            src_ld = load_ld_model(source_region, args.ld_models_base)
            if src_ld is not None and os.path.exists(original_png):
                base_dets = ld_infer(src_ld, original_png, args.ld_conf_thresh, args.ld_imgsz)
                baseline_cache[baseline_key] = {
                    "num": len(base_dets),
                    "classes": set(int(d["class"]) for d in base_dets),
                    "missing_model": False,
                    "missing_source": False,
                }
            else:
                baseline_cache[baseline_key] = {
                    "num": 0, "classes": set(),
                    "missing_model": src_ld is None,
                    "missing_source": not os.path.exists(original_png),
                }
        baseline = baseline_cache[baseline_key]

        # Load YOLO ground-truth labels (Level 2 verification) if available
        if baseline_key not in gt_cache:
            gt_path = os.path.join(args.source_data_dir, source_region,
                                   args.gt_labels_subdir, f"{stem}.txt")
            if os.path.exists(gt_path):
                gt_classes = set()
                with open(gt_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            gt_classes.add(int(parts[0]))
                gt_cache[baseline_key] = gt_classes
            else:
                gt_cache[baseline_key] = None
        gt_classes = gt_cache[baseline_key]

        ld_results = {}
        for region in predicted_regions:
            ld_model = load_ld_model(region, args.ld_models_base)
            if ld_model is None:
                ld_results[region] = {"skipped": True}
                continue
            detections = ld_infer(ld_model, png_path, args.ld_conf_thresh, args.ld_imgsz)
            sim_classes = set(int(d["class"]) for d in detections)
            # Only compare against baseline when the region is the source region (apples-to-apples)
            compare = None
            gt_compare = None
            if region == source_region and not baseline["missing_model"]:
                base_classes = baseline["classes"]
                tp = len(sim_classes & base_classes)
                fp = len(sim_classes - base_classes)    # landmarks found only in sim, not baseline
                fn = len(base_classes - sim_classes)    # landmarks in baseline, missed in sim
                compare = {
                    "baseline_num": baseline["num"],
                    "baseline_classes": len(base_classes),
                    "sim_num": len(detections),
                    "sim_classes": len(sim_classes),
                    "shared_classes": tp,
                    "lost_classes": fn,
                    "spurious_classes": fp,
                    "recall_vs_baseline": tp / len(base_classes) if base_classes else None,
                }
                # Level 2: verification against YOLO ground-truth labels
                if gt_classes is not None:
                    gt_tp_sim = len(sim_classes & gt_classes)
                    gt_tp_base = len(base_classes & gt_classes)
                    gt_compare = {
                        "gt_num_classes": len(gt_classes),
                        "sim_matches_gt": gt_tp_sim,
                        "baseline_matches_gt": gt_tp_base,
                        "sim_gt_recall": gt_tp_sim / len(gt_classes) if gt_classes else None,
                        "sim_gt_precision": gt_tp_sim / len(sim_classes) if sim_classes else None,
                        "baseline_gt_recall": gt_tp_base / len(gt_classes) if gt_classes else None,
                        "baseline_gt_precision": gt_tp_base / len(base_classes) if base_classes else None,
                    }
            ld_results[region] = {
                "num_detections": len(detections),
                "detection_classes": sorted(sim_classes),
                "vs_baseline": compare,
                "vs_ground_truth": gt_compare,
            }
            ld_totals[region] += len(detections)

        per_image.append({
            "source_region": source_region,
            "stem": stem,
            "angle": angle,
            "tint": tint,
            "png_path": png_path,
            "rc_predicted": predicted_regions,
            "rc_probs": {r: probs[i] for i, r in enumerate(region_ids)},
            "rc_correct": rc_correct,
            "baseline_num": baseline["num"],
            "baseline_classes": sorted(baseline["classes"]),
            "ld_results": ld_results,
        })

        if (i + 1) % 10 == 0 or i + 1 == len(entries):
            print(f"  [{i+1}/{len(entries)}] RC recall so far: {rc_hit}/{rc_total} = {100*rc_hit/rc_total:.1f}%")

    # Aggregate LD recall-vs-baseline AND vs-ground-truth by tint and by angle bucket
    def new_bucket():
        return {
            "recalls_base": [],
            "recalls_gt": [],
            "precisions_gt": [],
            "baseline_recalls_gt": [],
        }
    tint_stats = defaultdict(new_bucket)
    angle_stats = defaultdict(new_bucket)
    for e in per_image:
        src_res = e["ld_results"].get(e["source_region"], {})
        cmp_ = src_res.get("vs_baseline")
        gt_cmp = src_res.get("vs_ground_truth")
        if cmp_ is None or cmp_["recall_vs_baseline"] is None:
            continue
        for bucket_key, bucket in [(e["tint"], tint_stats), (int(round(e["angle"] / 5) * 5), angle_stats)]:
            bucket[bucket_key]["recalls_base"].append(cmp_["recall_vs_baseline"])
            if gt_cmp is not None:
                if gt_cmp["sim_gt_recall"] is not None:
                    bucket[bucket_key]["recalls_gt"].append(gt_cmp["sim_gt_recall"])
                if gt_cmp["sim_gt_precision"] is not None:
                    bucket[bucket_key]["precisions_gt"].append(gt_cmp["sim_gt_precision"])
                if gt_cmp["baseline_gt_recall"] is not None:
                    bucket[bucket_key]["baseline_recalls_gt"].append(gt_cmp["baseline_gt_recall"])

    def _mean(lst): return sum(lst) / len(lst) if lst else None

    def summarize_bucket(bucket):
        out = {}
        for k, v in bucket.items():
            out[str(k)] = {
                "num_images": len(v["recalls_base"]),
                "mean_recall_vs_baseline": _mean(v["recalls_base"]),
                "mean_recall_vs_gt": _mean(v["recalls_gt"]),
                "mean_precision_vs_gt": _mean(v["precisions_gt"]),
                "mean_baseline_recall_vs_gt": _mean(v["baseline_recalls_gt"]),
            }
        return out

    summary = {
        "num_images": len(entries),
        "rc_recall_on_source_region": rc_hit / rc_total if rc_total else 0.0,
        "rc_pred_thresh": args.rc_pred_thresh,
        "ld_conf_thresh": args.ld_conf_thresh,
        "ld_detections_per_region": dict(ld_totals),
        "ld_by_tint": summarize_bucket(tint_stats),
        "ld_by_angle_bucket": summarize_bucket(angle_stats),
        "per_image": per_image,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Pipeline] wrote {args.output_json}")
    print(f"[Pipeline] RC recall: {summary['rc_recall_on_source_region']*100:.1f}%")
    print(f"[Pipeline] LD total detections: {sum(ld_totals.values())} across {len(ld_totals)} regions")
    def _pct(x): return f"{x*100:.1f}%" if x is not None else "N/A"

    print("[Pipeline] By TINT (vs baseline / vs GT):")
    for k, v in summary["ld_by_tint"].items():
        print(f"  {k}: n={v['num_images']}  base_recall={_pct(v['mean_recall_vs_baseline'])}  "
              f"gt_recall={_pct(v['mean_recall_vs_gt'])}  gt_prec={_pct(v['mean_precision_vs_gt'])}  "
              f"(baseline_gt_recall={_pct(v['mean_baseline_recall_vs_gt'])})")
    print("[Pipeline] By ANGLE bucket (vs baseline / vs GT):")
    for k, v in sorted(summary["ld_by_angle_bucket"].items(), key=lambda kv: int(kv[0])):
        print(f"  ~{k}°: n={v['num_images']}  base_recall={_pct(v['mean_recall_vs_baseline'])}  "
              f"gt_recall={_pct(v['mean_recall_vs_gt'])}  gt_prec={_pct(v['mean_precision_vs_gt'])}")


if __name__ == "__main__":
    main()
