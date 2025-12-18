import argparse
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

# -----------------------------
# Image preprocessing
# -----------------------------
def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# -----------------------------
# Load ViT feature extractor
# -----------------------------
def load_vit(ckpt_path, num_classes, device):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )
    state = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.reset_classifier(0)  # feature extractor
    model.to(device).eval()
    return model

# -----------------------------
# Compute class prototypes
# -----------------------------
@torch.no_grad()
def compute_class_prototypes(real_root, model, transform, device):
    prototypes = {}

    class_dirs = sorted([d for d in real_root.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise RuntimeError(f"No class folders found in {real_root}")

    for cls_dir in class_dirs:
        feats = []
        imgs = list(cls_dir.glob("*.png"))
        if len(imgs) == 0:
            raise RuntimeError(f"No real images found in {cls_dir}")

        for img_path in tqdm(imgs, desc=f"Prototype class {cls_dir.name}"):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            f = model(x)
            f = F.normalize(f, dim=1)
            feats.append(f)

        feats = torch.cat(feats, dim=0)
        prototypes[int(cls_dir.name)] = feats.mean(dim=0, keepdim=True)

    return prototypes

# -----------------------------
# Filter synthetic images (J2)
# -----------------------------
@torch.no_grad()
def filter_synth(
    synth_root, out_root, prototypes, model,
    transform, device, keep_per_class
):
    out_root.mkdir(parents=True, exist_ok=True)

    for cls, proto in prototypes.items():
        cls_dir = synth_root / str(cls)
        if not cls_dir.exists():
            continue

        out_cls = out_root / str(cls)
        out_cls.mkdir(parents=True, exist_ok=True)

        scores = []
        paths = []

        for img_path in cls_dir.glob("*.png"):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            f = F.normalize(model(x), dim=1)
            sim = F.cosine_similarity(f, proto).item()
            scores.append(sim)
            paths.append(img_path)

        # Top-K selection
        idx = torch.tensor(scores).topk(min(keep_per_class, len(scores))).indices
        for i in idx:
            shutil.copy(paths[i], out_cls / paths[i].name)

        print(f"[Class {cls}] kept {len(idx)} synthetic samples")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("J2 Synthetic Filtering")
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--synth-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--keep-per-class", type=int, default=300)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    real_root = Path(args.real_root)
    synth_root = Path(args.synth_root)
    out_root = Path(args.out_root)

    assert real_root.exists(), f"REAL root not found: {real_root}"
    assert synth_root.exists(), f"SYNTH root not found: {synth_root}"

    transform = build_transform(args.img_size)
    model = load_vit(args.ckpt, args.num_classes, device)

    print("[INFO] Computing real class prototypes...")
    prototypes = compute_class_prototypes(real_root, model, transform, device)

    print("[INFO] Filtering synthetic images (J2)...")
    filter_synth(
        synth_root, out_root, prototypes,
        model, transform, device,
        args.keep_per_class
    )

    print("[DONE] J2-filtered synthetic dataset ready.")

if __name__ == "__main__":
    main()
