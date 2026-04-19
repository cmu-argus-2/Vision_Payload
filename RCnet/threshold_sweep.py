"""
Sweep classification threshold on validation set. Shape-agnostic via direct data loading.
"""
import argparse, os, sys, json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, '/home/sunac/Vision_Payload')
sys.path.insert(0, '/home/sunac/Vision_Payload/RCnet')

from vision_inference.region_classifier import RegionClassifier, ClassifierEfficient


def build_dataset(data_dir, regions, image_size, split='val', train_ratio=0.7, val_ratio=0.15, seed=42):
    # Mimic MGRSImageDataset file-collection logic for a given split
    import random
    all_files = []
    for r in regions:
        region_dir = os.path.join(data_dir, r)
        if not os.path.isdir(region_dir):
            continue
        for f in sorted(os.listdir(region_dir)):
            if not (f.endswith('.png') or f.endswith('.jpg')):
                continue
            if not f[3].isdigit():
                continue
            json_path = os.path.join(region_dir, f.rsplit('.', 1)[0] + '_mgrs_counts.json')
            if os.path.exists(json_path):
                all_files.append((os.path.join(region_dir, f), json_path))
    random.seed(seed)
    random.shuffle(all_files)
    n = len(all_files)
    tr = int(train_ratio * n); vl = int(val_ratio * n)
    if split == 'train': files = all_files[:tr]
    elif split == 'val': files = all_files[tr:tr+vl]
    else: files = all_files[tr+vl:]
    return files


def sigmoid_transform(x, k=58.8888, x0=0.25):
    import math
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_dir', default='/mnt/sda2/training/')
    ap.add_argument('--size', type=int, nargs=2, default=[224, 224], help='H W')
    args = ap.parse_args()

    regions = sorted(RegionClassifier.load_region_ids())
    region_idx = {r: i for i, r in enumerate(regions)}

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassifierEfficient().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    model.eval()

    # Val files
    files = build_dataset(args.data_dir, regions, args.size, 'val')
    print(f"Val images: {len(files)}")

    # Transform
    tfm = transforms.Compose([
        transforms.Resize(tuple(args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load broken files list
    import yaml
    broken = set()
    bpath = '/home/sunac/Vision_Payload/broken_files.yaml'
    if os.path.exists(bpath):
        with open(bpath) as f:
            broken = set(yaml.safe_load(f) or [])

    # Inference
    all_out, all_tgt = [], []
    skipped = 0
    with torch.no_grad():
        for img_path, json_path in files:
            if os.path.basename(img_path) in broken:
                skipped += 1
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                x = tfm(img).unsqueeze(0).to(device)
                out = model(x).cpu().squeeze(0)
            except Exception as e:
                skipped += 1
                continue
            # Build soft label
            counts = json.load(open(json_path))
            total = sum(counts.values())
            tgt = torch.zeros(len(regions))
            if total > 0:
                for z, c in counts.items():
                    if z in region_idx:
                        tgt[region_idx[z]] = sigmoid_transform(c / total)
            all_out.append(out); all_tgt.append(tgt)
    outputs = torch.stack(all_out)
    targets = torch.stack(all_tgt)

    for conv_name, target_bin in [
        ('STRICT (targets > 0.25 = one-of-4-regions)', targets > 0.25),
        ('SOFT (targets != 0 = any coverage)', targets != 0),
    ]:
        print(f"\n--- {conv_name} ---")
        print(f"{'Threshold':<11}{'F1':<9}{'P':<9}{'R':<9}")
        for t in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
            pred = outputs > t
            tp = (pred & target_bin).sum().item()
            fp = (pred & ~target_bin).sum().item()
            fn = (~pred & target_bin).sum().item()
            p = tp/(tp+fp) if (tp+fp)>0 else 0
            r = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*p*r/(p+r) if (p+r)>0 else 0
            print(f"{t:<11.3f}{f1*100:<9.2f}{p*100:<9.2f}{r*100:<9.2f}")


if __name__ == '__main__':
    main()
