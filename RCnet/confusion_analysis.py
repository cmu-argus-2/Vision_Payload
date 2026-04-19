"""Analyze which regions confuse with 16T (or any specified region)."""
import argparse, os, sys, json, math, yaml
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '/home/sunac/Vision_Payload')
sys.path.insert(0, '/home/sunac/Vision_Payload/RCnet')
from vision_inference.region_classifier import RegionClassifier, ClassifierEfficient


def sig(x, k=58.8888, x0=0.25):
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def build_val(data_dir, regions, seed=42, tr=0.7, vl=0.15):
    import random
    fs = []
    for r in regions:
        rd = os.path.join(data_dir, r)
        if not os.path.isdir(rd): continue
        for f in sorted(os.listdir(rd)):
            if not (f.endswith('.png') or f.endswith('.jpg')): continue
            if not f[3].isdigit(): continue
            jp = os.path.join(rd, f.rsplit('.', 1)[0] + '_mgrs_counts.json')
            if os.path.exists(jp):
                fs.append((os.path.join(rd, f), jp))
    random.seed(seed); random.shuffle(fs)
    n = len(fs); a = int(tr*n); b = int(vl*n)
    return fs[a:a+b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_dir', default='/mnt/sda2/training/')
    ap.add_argument('--target_region', default='16T')
    ap.add_argument('--strict_thresh', type=float, default=0.25)
    ap.add_argument('--pred_thresh', type=float, default=0.5)
    args = ap.parse_args()

    regions = sorted(RegionClassifier.load_region_ids())
    region_idx = {r: i for i, r in enumerate(regions)}
    target_idx = region_idx[args.target_region]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassifierEfficient().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    model.eval()

    broken = set()
    if os.path.exists('/home/sunac/Vision_Payload/broken_files.yaml'):
        broken = set(yaml.safe_load(open('/home/sunac/Vision_Payload/broken_files.yaml')) or [])

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    files = build_val(args.data_dir, regions)
    print(f"Val images: {len(files)}, analyzing region={args.target_region}")

    outs, tgts, paths = [], [], []
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
                        tgt[region_idx[z]] = sig(c / total)
            outs.append(out); tgts.append(tgt); paths.append(ip)

    outs = torch.stack(outs); tgts = torch.stack(tgts)

    # False positives for target_region: model predicts target but target is not really there
    fp_mask = (outs[:, target_idx] > args.pred_thresh) & ~(tgts[:, target_idx] > args.strict_thresh)
    print(f"\n{args.target_region} FALSE POSITIVES: {fp_mask.sum().item()}")
    print(f"When model wrongly says '{args.target_region}', what's ACTUALLY in those images?")
    # For each FP, tally which regions are actually present (target > strict)
    coocc = torch.zeros(len(regions))
    for i in fp_mask.nonzero().flatten():
        coocc += (tgts[i] > args.strict_thresh).float()
    n_fp = fp_mask.sum().item()
    print(f"{'Region':<8}{'Count':<8}{'%':<8}")
    for i, r in enumerate(regions):
        c = int(coocc[i].item())
        pct = 100*c/n_fp if n_fp else 0
        if c > 0: print(f"{r:<8}{c:<8}{pct:<8.1f}")

    # False negatives for target_region: model misses target
    fn_mask = ~(outs[:, target_idx] > args.pred_thresh) & (tgts[:, target_idx] > args.strict_thresh)
    print(f"\n{args.target_region} FALSE NEGATIVES: {fn_mask.sum().item()}")
    print(f"When model misses '{args.target_region}', what does it predict INSTEAD?")
    pred_instead = torch.zeros(len(regions))
    for i in fn_mask.nonzero().flatten():
        pred_instead += (outs[i] > args.pred_thresh).float()
    n_fn = fn_mask.sum().item()
    print(f"{'Region':<8}{'Count':<8}{'%':<8}")
    for i, r in enumerate(regions):
        c = int(pred_instead[i].item())
        pct = 100*c/n_fn if n_fn else 0
        if c > 0: print(f"{r:<8}{c:<8}{pct:<8.1f}")


if __name__ == '__main__':
    main()
