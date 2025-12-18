from pathlib import Path
import shutil
import random

REAL_ALL = Path("D:/EEG-Neurodiffusion/Topomaps_real/all")
OUT_ROOT = Path("D:/EEG-Neurodiffusion/Topomaps_real_LOSO")

NUM_SUBJECTS = 8   # choose based on your dataset
SEED = 42
random.seed(SEED)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

for cls_dir in REAL_ALL.iterdir():
    if not cls_dir.is_dir():
        continue

    images = list(cls_dir.glob("*.png"))
    random.shuffle(images)

    splits = [images[i::NUM_SUBJECTS] for i in range(NUM_SUBJECTS)]

    for s, imgs in enumerate(splits):
        subj_dir = OUT_ROOT / f"subject_{s+1:02d}" / cls_dir.name
        subj_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, subj_dir / img.name)

print("[DONE] LOSO dataset created.")
