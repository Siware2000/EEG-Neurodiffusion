import argparse
from pathlib import Path
import json
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def strip_module_prefix(state_dict):
    # handles DataParallel: "module.xxx"
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def load_ckpt_weights(ckpt_path: Path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            sd = obj["model_state_dict"]
        elif "model" in obj:
            sd = obj["model"]
        else:
            # could already be a state_dict
            sd = obj
    else:
        sd = obj
    sd = strip_module_prefix(sd)
    return sd

# ----------------------------
# Model (must match training)
# ----------------------------
def build_vit(num_classes: int):
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    return m

# ----------------------------
# Eval + collect details
# ----------------------------
@torch.no_grad()
def eval_collect(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    probs_all = []
    paths_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())

    # get file paths from ImageFolder (same order as loader if shuffle=False)
    # loader.dataset.samples = [(path,label), ...]
    paths_all = [p for (p, _) in loader.dataset.samples]

    return np.array(y_true), np.array(y_pred), np.array(probs_all), paths_all

def plot_confmat(cm, class_names, out_png: Path, out_pdf: Path, title: str):
    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # annotate
    thresh = cm.max() * 0.6 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}" if cm.dtype != int else str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close()

def plot_ablation(ablation_csv: Path, out_png: Path, out_pdf: Path):
    # Expect columns: Method, Accuracy, Macro-F1, Checkpoint
    rows = []
    with open(ablation_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    methods = [r["Method"] for r in rows]
    acc = [float(r["Accuracy"]) for r in rows]
    f1  = [float(r["Macro-F1"]) for r in rows]

    x = np.arange(len(methods))
    w = 0.38

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - w/2, acc, width=w, label="Accuracy")
    plt.bar(x + w/2, f1,  width=w, label="Macro-F1")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Ablation Results (Frozen)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close()

def save_top_grids(y_true, y_pred, probs, paths, out_dir: Path, img_size: int, k=16):
    """
    Saves:
    - top_correct_grid.(png/pdf)
    - top_wrong_grid.(png/pdf)
    based on confidence margin.
    """
    ensure_dir(out_dir)

    conf = probs[np.arange(len(probs)), y_pred]
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx   = np.where(y_true != y_pred)[0]

    # sort by confidence (descending)
    correct_sorted = correct_idx[np.argsort(-conf[correct_idx])]
    wrong_sorted   = wrong_idx[np.argsort(-conf[wrong_idx])]

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    def load_imgs(idxs):
        imgs = []
        for i in idxs[:k]:
            img = tfm(datasets.folder.default_loader(paths[i]))
            imgs.append(img)
        if len(imgs) == 0:
            return None
        return torch.stack(imgs, dim=0)

    top_correct = load_imgs(correct_sorted)
    top_wrong   = load_imgs(wrong_sorted)

    if top_correct is not None:
        grid = make_grid(top_correct, nrow=4, padding=2)
        save_image(grid, out_dir / "top_correct_grid.png")
    if top_wrong is not None:
        grid = make_grid(top_wrong, nrow=4, padding=2)
        save_image(grid, out_dir / "top_wrong_grid.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Real evaluation set root (ImageFolder: 0/1/2/3)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint to evaluate (best method or each method)")
    ap.add_argument("--out-dir", default="journal_figs", help="Output folder for figures")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ablation-csv", default=None, help="Path to eval_ablation_final/ablation_results.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Dataset (must match training normalization)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    ds = datasets.ImageFolder(args.data_root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model + load
    model = build_vit(num_classes=len(ds.classes)).to(device)
    sd = load_ckpt_weights(Path(args.ckpt), device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Eval
    y_true, y_pred, probs, paths = eval_collect(model, loader, device)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=ds.classes, digits=4, output_dict=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "summary.txt").write_text(
        f"checkpoint: {args.ckpt}\naccuracy: {acc:.4f}\nmacro_f1: {mf1:.4f}\n"
    )

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plot_confmat(cm, ds.classes,
                out_dir / "confusion_matrix.png",
                out_dir / "confusion_matrix.pdf",
                title="Confusion Matrix (Counts)")

    plot_confmat(cm_norm, ds.classes,
                out_dir / "confusion_matrix_normalized.png",
                out_dir / "confusion_matrix_normalized.pdf",
                title="Confusion Matrix (Row-normalized)")

    # Qualitative grids
    save_top_grids(y_true, y_pred, probs, paths, out_dir, img_size=args.img_size, k=16)

    # Ablation plot (if provided)
    if args.ablation_csv is not None:
        ab_csv = Path(args.ablation_csv)
        if ab_csv.exists():
            plot_ablation(
                ab_csv,
                out_dir / "ablation_bar.png",
                out_dir / "ablation_bar.pdf"
            )

    print("\n[DONE] Journal visuals exported to:", str(out_dir.resolve()))
    print(f"Accuracy={acc:.4f}  Macro-F1={mf1:.4f}")

if __name__ == "__main__":
    main()
