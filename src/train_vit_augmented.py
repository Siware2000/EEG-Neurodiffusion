import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import timm

from dataset_topomap import TopomapImageDataset


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)  # selects "GPU 0" etc.
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="D:/EEG-Neurodiffusion/topomaps_augmented/all",
                        help="Folder with class subfolders 0..C-1")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Windows-safe default: 0. (You can try 2 later.)")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=str, default="D:/EEG-Neurodiffusion/checkpoints_aug")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.gpu_id)
    print(f"[INFO] Using device: {device}")

    dataset = TopomapImageDataset(root_dir=args.data_root, img_size=args.img_size)

    # infer num_classes from folder structure
    num_classes = len(dataset.class_to_idx)
    print(f"[INFO] Dataset samples: {len(dataset)} | num_classes={num_classes}")

    # split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=False
    )

    # ViT (224 input)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "vit_aug_best.pt"

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"[INFO] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_acc,
                "num_classes": num_classes,
                "img_size": args.img_size,
            }, best_path)
            print(f"[SAVE] Best model saved to {best_path} (val_acc={best_acc:.4f})")

    print(f"\n[DONE] Best validation accuracy = {best_acc:.4f}")


if __name__ == "__main__":
    # Needed on Windows for DataLoader multiprocessing safety
    # (Even if num_workers=0, keep this guard)
    main()
