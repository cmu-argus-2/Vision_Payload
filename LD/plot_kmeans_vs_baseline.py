"""Bar chart: baseline vs k-means mAP50/P/R per region."""
import csv, os, re, glob
import matplotlib.pyplot as plt
import numpy as np

REGIONS = ["10S","10T","11R","12R","16T","17R","17T","18S",
           "32S","32T","33S","33T","52S","53S","54S","54T"]
RUNS_DIR = "/home/sunac/Vision_Payload/runs/detect"
LOGS = "/home/sunac/Vision_Payload/LD/logs"

def baseline(region):
    runs = sorted(glob.glob(f"{RUNS_DIR}/yolo_training_results_{region}/yolo_training_results_{region}_*/"))
    if not runs:
        return None
    csvp = os.path.join(runs[0], "results.csv")
    with open(csvp) as f:
        rows = list(csv.DictReader(f))
    if not rows: return None
    r = rows[-1]
    return (float(r["metrics/precision(B)"]),
            float(r["metrics/recall(B)"]),
            float(r["metrics/mAP50(B)"]))

def kmeans(region):
    candidates = [f"{LOGS}/train_kmeans100_{region}.log",
                  f"{LOGS}/train_33S_kmeans100.log" if region == "33S" else None]
    for path in filter(None, candidates):
        if not os.path.exists(path): continue
        with open(path, errors="ignore") as f:
            txt = f.read()
        matches = re.findall(r"\s+all\s+\d+\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", txt)
        if matches:
            P, R, mAP50, _ = matches[-1]
            return float(P), float(R), float(mAP50)
    return None

baselines = {r: baseline(r) for r in REGIONS}
kmeans_data = {r: kmeans(r) for r in REGIONS}

fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(len(REGIONS))
w = 0.13

bP = [baselines[r][0] if baselines[r] else 0 for r in REGIONS]
bR = [baselines[r][1] if baselines[r] else 0 for r in REGIONS]
bM = [baselines[r][2] if baselines[r] else 0 for r in REGIONS]
kP = [kmeans_data[r][0] if kmeans_data[r] else 0 for r in REGIONS]
kR = [kmeans_data[r][1] if kmeans_data[r] else 0 for r in REGIONS]
kM = [kmeans_data[r][2] if kmeans_data[r] else 0 for r in REGIONS]

ax.bar(x - 2.5*w, bP, w, label="baseline P", color="#a6cee3")
ax.bar(x - 1.5*w, bR, w, label="baseline R", color="#1f78b4")
ax.bar(x - 0.5*w, bM, w, label="baseline mAP50", color="#08306b")
ax.bar(x + 0.5*w, kP, w, label="kmeans P", color="#fdbf6f")
ax.bar(x + 1.5*w, kR, w, label="kmeans R", color="#ff7f00")
ax.bar(x + 2.5*w, kM, w, label="kmeans mAP50", color="#b15928")

ax.set_xticks(x)
ax.set_xticklabels(REGIONS)
ax.set_ylim(0, 1.05)
ax.set_ylabel("score")
ax.set_title("Baseline (random-angle val, all 16 regions) vs K-means re-selection (8-fixed-angle val, 7 regions only)")
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.9, color="gray", lw=0.5, ls="--")

# annotate kmeans mAP50 above bars where present
for i, r in enumerate(REGIONS):
    if kmeans_data[r]:
        ax.text(x[i] + 2.5*w, kM[i] + 0.01, f"{kM[i]:.2f}", ha="center", fontsize=7, color="#b15928")

out = f"{LOGS}/16regions_baseline_vs_kmeans_PR.png"
plt.tight_layout()
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")

print("\nRegion | base mAP50 | kmeans mAP50 | delta")
for r in REGIONS:
    b = baselines[r][2] if baselines[r] else None
    k = kmeans_data[r][2] if kmeans_data[r] else None
    if b is not None and k is not None:
        print(f"  {r:5s} | {b:.3f}      | {k:.3f}        | {k-b:+.3f}")
    elif b is not None:
        print(f"  {r:5s} | {b:.3f}      | -            |")
