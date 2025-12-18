# src/generate_topomaps_all.py
"""
Generate topomap-like 2D grids for SEED-IV, STEW, and CLAS
using a uniform pipeline:

  SEED-IV + CLAS: time-series -> bandpower per channel (5 bands)
                  -> 2D grids -> RGB PNG + NPY

  STEW: precomputed feature vectors -> 2D grids
        -> grayscale-as-RGB PNG + NPY

Output structure (on your PortableSSD F:):
  F:/EEG-Neurodiffusion/topomaps/
    seed_iv/0,1,2,3
    stew/0,1
    clas/0,1,2
    all/0,1,2,3  # combined multi-dataset corpus (PNG only)
"""

import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from scipy.signal import welch

# === Import your existing loaders ===
from preprocess_seed_iv import load_and_preprocess_seed_iv
from stew_pipeline import collect_stew_trials
from load_clas_dataset import build_clas_dataset

# -------------------------------
# 1. Bandpower utilities
# -------------------------------

# (low-freq) delta, theta, alpha, beta, gamma
BANDS: List[Tuple[float, float]] = [
    (1.0, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 30.0),  # beta
    (30.0, 45.0),  # gamma
]

# For RGB, we’ll use (theta, alpha, beta) = bands 1,2,3
RGB_BAND_IDXS = (1, 2, 3)


def bandpower_timeseries(
    x: np.ndarray,
    sfreq: float,
    bands: List[Tuple[float, float]] = BANDS,
) -> np.ndarray:
    """
    Compute bandpower for a (channels, time) or (time,) array.

    Returns shape:
      - if x shape (C, T): (C, len(bands))
      - if x shape (T,):   (len(bands),)
    """
    x = np.asarray(x)

    if x.ndim == 1:
        x = x[np.newaxis, :]

    n_channels, n_samples = x.shape

    freqs, psd = welch(x, sfreq, nperseg=min(1024, n_samples))

    bandpowers = np.zeros((n_channels, len(bands)), dtype=np.float32)
    for bi, (fmin, fmax) in enumerate(bands):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        bandpowers[:, bi] = np.trapz(psd[:, idx], freqs[idx], axis=-1)

    return bandpowers  # (C, num_bands)


# -------------------------------
# 2. 1D features → 2D grid
# -------------------------------

def features_to_grid(features_1d: np.ndarray) -> np.ndarray:
    """
    Map a vector of length N into a square grid (H, W) by row-major fill,
    zero-padding any remaining cells.

    features_1d: shape (N,)
    returns: grid of shape (H, W)
    """
    features_1d = np.asarray(features_1d, dtype=np.float32).ravel()
    N = features_1d.shape[0]
    grid_size = int(math.ceil(math.sqrt(N)))  # smallest square that fits N
    H = W = grid_size

    grid = np.zeros((H * W,), dtype=np.float32)
    grid[:N] = features_1d
    grid = grid.reshape(H, W)
    return grid


def normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    _min = x.min()
    _max = x.max()
    if _max - _min < eps:
        return np.zeros_like(x)
    return (x - _min) / (_max - _min)


def bands_to_rgb_grid(bandpowers: np.ndarray) -> np.ndarray:
    """
    bandpowers: (num_channels, num_bands)

    Steps:
      1. Split into per-band 1D vectors (over channels).
      2. Map each band vector to 2D grid with same (H, W).
      3. Select 3 bands for RGB (theta, alpha, beta).
    Returns:
      rgb_image: (H, W, 3) float32 in [0,1]
    """
    num_channels, num_bands = bandpowers.shape
    assert num_bands >= 3, "Need at least 3 bands for RGB."

    grids = []
    base_H, base_W = None, None
    for b_idx in range(num_bands):
        vec = bandpowers[:, b_idx]  # (num_channels,)
        grid = features_to_grid(vec)  # (H, W)
        if base_H is None:
            base_H, base_W = grid.shape
        else:
            assert grid.shape == (base_H, base_W)
        grids.append(grid)

    r = grids[RGB_BAND_IDXS[0]]
    g = grids[RGB_BAND_IDXS[1]]
    b = grids[RGB_BAND_IDXS[2]]

    r = normalize_minmax(r)
    g = normalize_minmax(g)
    b = normalize_minmax(b)

    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)
    return rgb.astype(np.float32)


def save_rgb_array(rgb: np.ndarray, path: Path, img_size: int = 224):
    """
    rgb: (H,W,3) float32 in [0,1]
    path: where to save
    """
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


# -------------------------------
# 3. Dataset-specific processors
# -------------------------------

def process_seed_iv(out_root: Path, sfreq: float = 200.0, img_size: int = 224):
    print("=== Generating topomaps for SEED-IV ===")
    X, y = load_and_preprocess_seed_iv()  # (n_trials, C, T), (n_trials,)
    print("SEED-IV shapes:", X.shape, y.shape)

    for idx in range(X.shape[0]):
        trial = X[idx]  # (C, T)
        label = int(y[idx])

        bp = bandpower_timeseries(trial, sfreq=sfreq)  # (C, num_bands)
        rgb = bands_to_rgb_grid(bp)

        # PNG
        out_png = out_root / "seed_iv" / str(label) / f"seediv_{idx:05d}.png"
        save_rgb_array(rgb, out_png, img_size=img_size)

        # NPY (store bandpower grids per band)
        out_npy = out_root / "seed_iv" / str(label) / f"seediv_{idx:05d}.npy"
        np.save(out_npy, bp.astype(np.float32))

    print("SEED-IV topomaps done.")


def process_stew(
    out_root: Path,
    stew_root: str = r"D:\EEG-Neurodiffusion\Data Set\STEW Dataset\STEW Dataset",
    img_size: int = 224,
):
    """
    STEW: we already have per-trial feature vectors from collect_stew_trials.
    We just map each feature vector to a 2D grid and replicate as RGB.
    """
    print("=== Generating topomaps for STEW ===")

    # ✅ FIX: pass stew_root into the loader
    X_stew, y_stew, meta = collect_stew_trials(stew_root)
    print("STEW shapes:", X_stew.shape, y_stew.shape)

    for idx in range(X_stew.shape[0]):
        feat = X_stew[idx].ravel()
        grid = features_to_grid(feat)           # (H,W)
        grid_norm = normalize_minmax(grid)

        # Use grayscale grid replicated to RGB
        rgb = np.stack([grid_norm, grid_norm, grid_norm], axis=-1).astype(
            np.float32
        )

        label = int(y_stew[idx])

        out_dir = out_root / "stew" / str(label)
        out_dir.mkdir(parents=True, exist_ok=True)

        # PNG
        out_png = out_dir / f"stew_{idx:05d}.png"
        save_rgb_array(rgb, out_png, img_size=img_size)

        # NPY (raw grid)
        out_npy = out_dir / f"stew_{idx:05d}.npy"
        np.save(out_npy, grid.astype(np.float32))

    print("STEW topomaps done.")


def process_clas(out_root: Path, sfreq: float = 200.0, img_size: int = 224):
    """
    CLAS: ECG + GSR + PPG time-series, shape (n_trials, T, C).
    We:
      - transpose to (C,T),
      - compute bandpower per channel,
      - convert to RGB grid,
      - save PNG + NPY.
    """
    print("=== Generating topomaps for CLAS ===")

    # ✅ FIX: build_clas_dataset returns only (X, y)
    X_clas, y_clas = build_clas_dataset()  # (N, T, C), (N,)
    print("CLAS shapes:", X_clas.shape, y_clas.shape)

    for idx in range(X_clas.shape[0]):
        trial_tc = X_clas[idx]          # (T, C)
        trial = trial_tc.T              # -> (C, T)
        label = int(y_clas[idx])

        bp = bandpower_timeseries(trial, sfreq=sfreq)  # (C, num_bands)
        rgb = bands_to_rgb_grid(bp)

        out_dir = out_root / "clas" / str(label)
        out_dir.mkdir(parents=True, exist_ok=True)

        # PNG
        out_png = out_dir / f"clas_{idx:05d}.png"
        save_rgb_array(rgb, out_png, img_size=img_size)

        # NPY (bandpowers)
        out_npy = out_dir / f"clas_{idx:05d}.npy"
        np.save(out_npy, bp.astype(np.float32))

    print("CLAS topomaps done.")


# -------------------------------
# 4. Combined folder "all"
# -------------------------------

def populate_combined_root(out_root: Path):
    """
    Create out_root / 'all' / {0,1,2,3} containing copies of all PNGs
    from seed_iv, stew, clas.
    """
    print("=== Building combined 'all' corpus ===")
    for dataset_name in ["seed_iv", "stew", "clas"]:
        ds_root = out_root / dataset_name
        if not ds_root.exists():
            print(f"[WARN] Dataset folder {ds_root} does not exist, skipping.")
            continue

        for label_dir in sorted(ds_root.glob("*")):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            target_dir = out_root / "all" / label
            target_dir.mkdir(parents=True, exist_ok=True)

            for img_path in label_dir.glob("*.png"):
                target_path = target_dir / f"{dataset_name}_{img_path.name}"
                if not target_path.exists():
                    with open(img_path, "rb") as src_f, open(target_path, "wb") as dst_f:
                        dst_f.write(src_f.read())

    print("Combined 'all' topomap folder built.")


# -------------------------------
# 5. Main
# -------------------------------

def main():
    # Your topomap root on F: (matches the screenshot)
    out_root = Path(r"D:\EEG-Neurodiffusion\topomaps")

    # 1) Per-dataset topomaps
    process_seed_iv(out_root)
    process_stew(out_root)
    process_clas(out_root)

    # 2) Combined multi-dataset corpus (PNG only)
    populate_combined_root(out_root)
    print("All topomaps generated successfully.")


if __name__ == "__main__":
    main()
