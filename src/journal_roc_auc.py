# src/journal_roc_auc.py
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def build_vit(num_classes: int):
    """Must match your training (torchvision ViT)."""
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Real test set root (ImageFolder).")
    p.add_argument("--ckpt", required=True, help="ViT checkpoint path.")
    p.add_argument("--out-dir", required=True, help="Output folder.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(args.data_root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(ds.classes)
    model = build_vit(num_classes=num_classes).to(device)

    state = torch.load(args.ckpt, map_location=device)
    # support both raw state_dict and dict-wrapped
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)

    logits, y = collect_logits_and_labels(model, loader, device)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    # One-vs-rest
    y_bin = label_binarize(y, classes=list(range(num_classes)))

    plt.figure()
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves (ViT)")
    plt.legend(loc="lower right")

    out_path = out_dir / "roc_ovr.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
