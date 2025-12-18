# src/eeg_topomap_dataset.py

from pathlib import Path
from typing import List, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class TopomapImageDataset(Dataset):
    """
    Generic PNG dataset reader for:
        root_dir / <dataset_name> / <label> / *.png

    Example:
        root_dir = "D:/EEG-Neurodiffusion/Topomaps"
        dataset  = "all" | "seed_iv" | "stew" | "clas"
    """
    def __init__(self, root_dir: str, dataset: str = "all", image_size: int = 224):
        self.root_dir = Path(root_dir)
        self.dataset = dataset
        self.image_size = image_size

        ds_root = self.root_dir / dataset
        if not ds_root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {ds_root}")

        self.samples: List[Tuple[Path, int]] = []

        for label_dir in sorted(ds_root.iterdir()):
            if not label_dir.is_dir():
                continue
            # label folders are assumed to be "0", "1", "2", ...
            try:
                label = int(label_dir.name)
            except ValueError:
                # ignore non-numeric dirs
                continue

            for img_path in label_dir.glob("*.png"):
                self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No PNG files found under {ds_root}")

        self.num_classes = len({lbl for _, lbl in self.samples})
        print(
            f"[TopomapImageDataset] dataset={dataset}, "
            f"samples={len(self.samples)}, num_classes={self.num_classes}"
        )

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),  # -> [0,1], shape (3,H,W)
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label
