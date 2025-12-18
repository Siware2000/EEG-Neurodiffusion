# src/generate_topomaps_seed_iv.py

import os
import numpy as np

import mne  # <-- THIS fixes "name 'mne' is not defined"
import matplotlib
matplotlib.use("Agg")  # for non-interactive saving
import matplotlib.pyplot as plt

from preprocess_seed_iv import load_and_preprocess_seed_iv

# Where to save SEED-IV topomaps (change if you like)
DEFAULT_OUT_DIR = r"D:\EEG-Neurodiffusion\Topomaps\SEED_IV"

def _make_seed_iv_info(n_channels: int, sfreq: float = 250.0):
    """
    Create an MNE Info object + montage for SEED-IV channels.
    We don't know the *true* names from the .mat file, so we
    map channel indices to the first n_channels in the standard_1020 montage.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    all_ch_names = montage.ch_names

    if len(all_ch_names) < n_channels:
        raise RuntimeError(
            f"Montage only has {len(all_ch_names)} channels but SEED-IV has {n_channels}"
        )

    # Use the first n_channels positions
    use_ch_names = all_ch_names[:n_channels]
    montage_subset = montage.copy().pick_channels(use_ch_names)

    info = mne.create_info(
        ch_names=use_ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )
    info.set_montage(montage_subset)
    return info


def generate_topomaps_seed_iv(
    out_dir: str = DEFAULT_OUT_DIR,
    overwrite: bool = False,
):
    """
    1) Load SEED-IV EEG (X, y) from preprocess_seed_iv.py
    2) For each trial, compute RMS over time per channel
    3) Save a single topomap PNG per trial, in class-wise folders:
           out_dir/class_0/seed_iv_trial_0000.png, etc.
    """
    print("=== [SEED-IV] Loading preprocessed EEG ===")
    X, y = load_and_preprocess_seed_iv()  # X: (N, 62, T), y: (N,)
    n_trials, n_channels, n_samples = X.shape
    print(f"Shape X: {X.shape}, y: {y.shape}, unique labels: {np.unique(y)}")

    os.makedirs(out_dir, exist_ok=True)

    # MNE Info with a pseudo-10–20 layout for those 62 channels
    info = _make_seed_iv_info(n_channels=n_channels, sfreq=250.0)

    # For each label, make a subfolder class_0, class_1, class_2
    for cls in np.unique(y):
        cls_dir = os.path.join(out_dir, f"class_{int(cls)}")
        os.makedirs(cls_dir, exist_ok=True)

    print("=== [SEED-IV] Generating topomaps ===")
    for idx in range(n_trials):
        trial = X[idx]          # (n_channels, T)
        label = int(y[idx])

        # Simple feature per channel: RMS over time
        vals = np.sqrt(np.mean(trial ** 2, axis=-1))  # (n_channels,)

        fig, ax = plt.subplots(figsize=(4, 3))
        mne.viz.plot_topomap(vals, info, axes=ax, show=False)
        ax.set_title(f"SEED-IV Trial {idx} (class={label})")

        cls_dir = os.path.join(out_dir, f"class_{label}")
        fname = os.path.join(cls_dir, f"seed_iv_trial_{idx:04d}.png")
        fig.savefig(fname, dpi=120, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

        if (idx + 1) % 100 == 0:
            print(f"  saved {idx + 1}/{n_trials} topomaps")

    print("=== [SEED-IV] Done. Topomaps saved in:", out_dir)


if __name__ == "__main__":
    generate_topomaps_seed_iv()
