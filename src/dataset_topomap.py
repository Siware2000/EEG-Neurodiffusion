# src/dataset_topomap.py

import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TopomapImageDataset(Dataset):
    """
    Expects directory structure like:
    root/
      class0/
         img1.png
         img2.png
      class1/
         img3.png
         ...
    """

    def __init__(self, root_dir: str, img_size: int = 128):
        self.root_dir = root_dir
        self.samples = []  # (filepath, label)

        class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                    )

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # ViT expects roughly ImageNet-like normalization
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
