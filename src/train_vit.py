# src/train_vit.py

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import timm
from tqdm import tqdm

from dataset_topomap import TopomapImageDataset


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


def train_vit_from_folder(
    topomap_root: str,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-5,
    num_workers: int = 0,
    save_path: str = "models/vit_topomap.pt",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = TopomapImageDataset(topomap_root, img_size=224)
    num_classes = len(dataset.class_to_idx)
    print("Found classes:", dataset.class_to_idx)

    # 80/20 split
    val_len = max(1, int(0.2 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = ViTClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_ds)

        # ---- validation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1, total)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
        },
        save_path,
    )
    print("Model saved to:", save_path)
