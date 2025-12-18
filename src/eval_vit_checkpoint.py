import argparse
from pathlib import Path
import json
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score


# -------------------------
# Build EXACT ViT used in training
# -------------------------
def build_vit(num_classes: int):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


# -------------------------
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    ds = datasets.ImageFolder(args.data_root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = build_vit(num_classes=len(ds.classes)).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    acc, f1 = evaluate(model, loader, device)

    # Save metrics
    metrics = {
        "checkpoint": args.ckpt,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4)
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("\n=== EVALUATION RESULT ===")
    print(metrics)


if __name__ == "__main__":
    main()
