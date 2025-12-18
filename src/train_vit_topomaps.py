import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm


def get_dataloaders(data_dir: Path,
                    img_size: int = 224,
                    batch_size: int = 32,
                    val_split: float = 0.2):

    # Standard ViT transforms (ImageNet style)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_tf)
    num_classes = len(full_dataset.classes)
    print(f"[INFO] Found {len(full_dataset)} images, classes = {full_dataset.classes}")

    # Train/val split
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Important: use val transforms for val set
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="ViT training on EEG topomaps")
    parser.add_argument("--topomaps-root", type=str,
                        default="D:/EEG-Neurodiffusion/topomaps",
                        help="Root folder containing seed_iv/stew/clas/all")
    parser.add_argument("--dataset", "-d", type=str,
                        choices=["seed_iv", "stew", "clas", "all"],
                        default="all",
                        help="Which dataset folder to use")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224",
                        help="timm model name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str,
                        default="D:/EEG-Neurodiffusion/checkpoints",
                        help="Where to save trained models")
    args = parser.parse_args()

    data_dir = Path(args.topomaps_root) / args.dataset
    assert data_dir.exists(), f"{data_dir} does not exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir, img_size=args.img_size, batch_size=args.batch_size
    )

    print(f"[INFO] num_classes = {num_classes}")

    # Create ViT model
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"vit_{args.dataset}.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_to_idx": train_loader.dataset.dataset.class_to_idx,
                "args": vars(args),
            }, ckpt_path)
            print(f"[INFO] New best model saved to {ckpt_path} (val_acc={val_acc:.4f})")

    print(f"[DONE] Training finished. Best val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
