import shutil
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================
REAL_ROOT = Path("D:/EEG-Neurodiffusion/topomaps/all")
SYN_ROOT  = Path("D:/EEG-Neurodiffusion/Topomaps_synth/all")
OUT_ROOT  = Path("D:/EEG-Neurodiffusion/topomaps_augmented/all")

NUM_CLASSES = 4   # all = 4 classes (0,1,2,3)

# =========================
# MERGE FUNCTION
# =========================
def merge_real_and_synth():
    for c in range(NUM_CLASSES):
        real_dir = REAL_ROOT / str(c)
        synth_dir = SYN_ROOT / str(c)
        out_dir = OUT_ROOT / str(c)

        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Copy REAL images ----
        for f in tqdm(list(real_dir.glob("*.png")), desc=f"Class {c} | Real"):
            dst = out_dir / f"real_{f.name}"
            shutil.copyfile(f, dst)

        # ---- Copy SYNTHETIC images ----
        for f in tqdm(list(synth_dir.glob("*.png")), desc=f"Class {c} | Synth"):
            dst = out_dir / f"synth_{f.name}"
            shutil.copyfile(f, dst)

        print(f"[OK] Class {c}: merged into {out_dir}")

    print("\n[DONE] Real + Diffusion synthetic dataset ready!")


if __name__ == "__main__":
    merge_real_and_synth()
