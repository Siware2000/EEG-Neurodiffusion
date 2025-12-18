import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# -----------------------
def build_vit(num_classes):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs_all, y_all = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        probs_all.append(probs.cpu().numpy())
        y_all.append(y.numpy())

    return np.vstack(probs_all), np.concatenate(y_all)


# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="journal_figs/roc")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    ds = datasets.ImageFolder(args.data_root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_vit(len(ds.classes)).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    probs, y = collect_probs(model, loader, device)
    y_bin = label_binarize(y, classes=list(range(len(ds.classes))))

    plt.figure(figsize=(7, 6))
    for i, cls in enumerate(ds.classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.grid(True)

    plt.savefig(out_dir / "roc_ovr.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "roc_ovr.pdf", dpi=600, bbox_inches="tight")
    plt.close()

    print("[DONE] ROC/AUC curves saved to:", out_dir)


if __name__ == "__main__":
    main()
