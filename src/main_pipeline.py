# src/main_pipeline.py

import os
from glob import glob

from load_and_preprocess import load_raw_auto, basic_preprocess, debug_info
from generate_topomaps import generate_topomaps_for_epochs
from train_vit import train_vit_from_folder

# Adjust these paths:
DATA_ROOT = r"D:\EEG-Neurodiffusion\Data Set"
TOPOMAP_ROOT = r"D:\EEG-Neurodiffusion\topomaps"


def find_example_file() -> str:
    """
    Pick the first .edf (or other) file we find in any dataset folder.
    Change the extension if your dataset is different.
    """
    patterns = [
        os.path.join(DATA_ROOT, "**", "*.edf"),
        os.path.join(DATA_ROOT, "**", "*.vhdr"),
        os.path.join(DATA_ROOT, "**", "*.bdf"),
        os.path.join(DATA_ROOT, "**", "*.fif"),
    ]
    for pat in patterns:
        files = glob(pat, recursive=True)
        if files:
            print("Using file:", files[0])
            return files[0]
    raise FileNotFoundError("No EEG file (.edf/.vhdr/.bdf/.fif) found. "
                            "Adjust patterns in find_example_file().")


def run_pipeline():
    # 1. pick a file
    raw_file = find_example_file()

    # 2. load & preprocess
    raw = load_raw_auto(raw_file)
    epochs = basic_preprocess(raw, l_freq=1.0, h_freq=45.0, epoch_length=2.0)
    debug_info(epochs)

    # 3. generate topomaps (currently all label = 0 => 'class0')
    generate_topomaps_for_epochs(
        epochs,
        out_dir=TOPOMAP_ROOT,
        label=0,
    )
    print("Topomaps generated in:", TOPOMAP_ROOT)

    # 4. Train a simple ViT on these topomaps
    train_vit_from_folder(TOPOMAP_ROOT, epochs=1)  # increase epochs later


if __name__ == "__main__":
    run_pipeline()
