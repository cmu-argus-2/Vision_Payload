"""Per-class precision/recall/F1 on validation set."""
import argparse, os, sys, json, math, yaml
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '/home/sunac/Vision_Payload')
sys.path.insert(0, '/home/sunac/Vision_Payload/RCnet')
from vision_inference.region_classifier import RegionClassifier, ClassifierEfficient


def sigmoid_transform(x, k=58.8888, x0=0.25):
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def build_val_files(data_dir, regions, seed=42, tr=0.7, vl=0.15):
    import random
    all_files = []
    for r in regions:
        rd = os.path.join(data_dir, r)
        if not os.path.isdir(rd): continue
        for f in sorted(os.listdir(rd)):
            if not (f.endswith('.png') or f.endswith('.jpg')): continue
            if not f[3].isdigit(): continue
            jp = os.path.join(rd, f.rsplit('.', 1)[0] + '_mgrs_counts.json')
            if os.path.exists(jp):
                all_files.append((os.path.join(rd, f), jp))
    random.seed(seed); random.shuffle(all_files)
    n = len(all_files); a = int(tr*n); b = int(vl*n)
    return all_files[a:a+b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_dir', default='/mnt/sda2/training/')
    ap.add_argument('--size', type=int, nargs=2, default=[224, 224])
    ap.add_argument('--strict_thresh', type=float, default=0.25)
    ap.add_argument('--pred_thresh', type=float, default=0.5)
    args = ap.parse_args()

    regions = sorted(RegionClassifier.load_region_ids())
    region_idx = {r: i for i, r in enumerate(regions)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassifierEfficient().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    model.eval()

    broken = set()
    bp = '/home/sunac/Vision_Payload/broken_files.yaml'
    if os.path.exists(bp):
        broken = set(yaml.safe_load(open(bp)) or [])

    tfm = transforms.Compose([
        transforms.Resize(tuple(args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    files = build_val_files(args.data_dir, regions)
    print(f"Val images: {len(files)}")

    all_out, all_tgt = [], []
    with torch.no_grad():
        for ip, jp in files:
            if os.path.basename(ip) in broken: continue
            try:
                img = Image.open(ip).convert('RGB')
                x = tfm(img).unsqueeze(0).to(device)
                out = model(x).cpu().squeeze(0)
            except Exception:
                continue
            counts = json.load(open(jp))
            total = sum(counts.values())
            tgt = torch.zeros(len(regions))
            if total > 0:
                for z, c in counts.items():
                    if z in region_idx:
                        tgt[region_idx[z]] = sigmoid_transform(c / total)
            all_out.append(out); all_tgt.append(tgt)

    outputs = torch.stack(all_out)
    targets = torch.stack(all_tgt)
    target_bin = targets > args.strict_thresh
    pred = outputs > args.pred_thresh

    print(f"\nPer-class (STRICT label>{args.strict_thresh}, pred>{args.pred_thresh}):")
    print(f"{'Region':<8}{'F1':<8}{'Prec':<8}{'Recall':<8}{'Support':<8}")
    print("-" * 40)
    results = []
    for i, r in enumerate(regions):
        tp = (pred[:, i] & target_bin[:, i]).sum().item()
        fp = (pred[:, i] & ~target_bin[:, i]).sum().item()
        fn = (~pred[:, i] & target_bin[:, i]).sum().item()
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r_m = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r_m/(p+r_m) if (p+r_m)>0 else 0
        support = target_bin[:, i].sum().item()
        results.append((r, f1, p, r_m, support))

    # sorted by F1 ascending (weakest first)
    for r, f1, p, r_m, support in sorted(results, key=lambda x: x[1]):
        print(f"{r:<8}{f1*100:<8.2f}{p*100:<8.2f}{r_m*100:<8.2f}{support:<8}")


if __name__ == '__main__':
    main()
