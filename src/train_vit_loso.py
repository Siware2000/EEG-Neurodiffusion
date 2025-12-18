import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def build_vit(num_classes):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    subjects = sorted([d for d in Path(args.data_root).iterdir() if d.is_dir()])

    all_acc, all_f1, cms = [], [], []

    for test_subj in subjects:
        print(f"\n[LOSO] Test subject: {test_subj.name}")

        train_dirs = [d for d in subjects if d != test_subj]

        train_ds = torch.utils.data.ConcatDataset([
            datasets.ImageFolder(d, transform=tfm) for d in train_dirs
        ])
        test_ds = datasets.ImageFolder(test_subj, transform=tfm)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = build_vit(num_classes=len(test_ds.classes)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        ce = nn.CrossEntropyLoss()

        # --- Train ---
        model.train()
        for _ in range(args.epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = ce(model(x), y)
                loss.backward()
                opt.step()

        # --- Test ---
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                p = model(x).argmax(1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(p.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        cm  = confusion_matrix(y_true, y_pred)

        all_acc.append(acc)
        all_f1.append(f1)
        cms.append(cm)

        print(f"Acc={acc:.4f}, Macro-F1={f1:.4f}")

    print("\n=== LOSO FINAL RESULTS ===")
    print(f"Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Macro-F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

    mean_cm = np.mean(cms, axis=0)
    print("Mean Confusion Matrix:\n", mean_cm)

if __name__ == "__main__":
    main()
