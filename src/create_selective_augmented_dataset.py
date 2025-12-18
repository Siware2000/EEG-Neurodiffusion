# src/create_selective_augmented_dataset.py

import random
import shutil
from pathlib import Path

# ======================================================
# USER CONFIG (EDIT ONLY IF PATHS CHANGE)
# ======================================================

# Real topomaps (generated earlier from SEED-IV, STEW, CLAS)
REAL_ROOT = Path("D:/EEG-Neurodiffusion/Topomaps/all")

# Synthetic topomaps (generated via diffusion sampling)
SYNTH_ROOT = Path("D:/EEG-Neurodiffusion/Topomaps_synth/all")

# Output augmented dataset
OUT_ROOT = Path("D:/EEG-Neurodiffusion/Topomaps_augmented_v2")

# Classes to augment (minority / weak classes)
TARGET_CLASSES = [0, 3]

# Fraction of synthetic samples to add (relative to real count)
SYNTH_RATIO = 0.25   # 25%

# Reproducibility
SEED = 42
random.seed(SEED)

# ======================================================
# SAFETY CHECKS
# ======================================================
assert REAL_ROOT.exists(), f"REAL_ROOT not found: {REAL_ROOT}"
assert SYNTH_ROOT.exists(), f"SYNTH_ROOT not found: {SYNTH_ROOT}"

# ======================================================
# MAIN LOGIC
# ======================================================

def main():
    print("=== Creating selective augmented dataset ===")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for cls_dir in sorted(REAL_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue

        cls = int(cls_dir.name)
        out_cls_dir = OUT_ROOT / str(cls)
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # 1) Copy ALL real images
        # --------------------------------------------------
        real_images = list(cls_dir.glob("*.png"))
        for img in real_images:
            shutil.copy(img, out_cls_dir / img.name)

        print(f"[Class {cls}] Real images copied: {len(real_images)}")

        # --------------------------------------------------
        # 2) Selectively add synthetic images (minority only)
        # --------------------------------------------------
        if cls in TARGET_CLASSES:
            synth_cls_dir = SYNTH_ROOT / str(cls)

            if synth_cls_dir.exists():
                synth_images = list(synth_cls_dir.glob("*.png"))

                n_add = int(len(real_images) * SYNTH_RATIO)
                n_add = min(n_add, len(synth_images))

                if n_add > 0:
                    selected = random.sample(synth_images, n_add)
                    for img in selected:
                        shutil.copy(img, out_cls_dir / img.name)

                print(f"[Class {cls}] Synthetic added: {n_add}")
            else:
                print(f"[Class {cls}] No synthetic folder found")

        total_images += len(list(out_cls_dir.glob("*.png")))

    # ======================================================
    # SUMMARY
    # ======================================================
    print("\n=== AUGMENTED DATASET SUMMARY ===")
    total = 0
    for cls_dir in sorted(OUT_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        n = len(list(cls_dir.glob("*.png")))
        total += n
        print(f"Class {cls_dir.name}: {n} images")

    print(f"TOTAL images: {total}")
    print("[DONE] Real + selective diffusion dataset ready!")

# ======================================================
# ENTRY POINT (Windows-safe)
# ======================================================
if __name__ == "__main__":
    main()
