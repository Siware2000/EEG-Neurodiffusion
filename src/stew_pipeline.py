import os
import glob
import numpy as np
import pandas as pd

from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

STEW_ROOT = r"D:\EEG-Neurodiffusion\Data Set\STEW Dataset\STEW Dataset"
FS = 128  # STEW sampling rate (Hz) - check ReadMe if you want to change


# EEG frequency bands (Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def load_eeg_file(path: str) -> np.ndarray:
    """
    Load a single STEW text EEG file.
    Each file is usually: samples x channels (whitespace- or tab-separated).

    Returns:
        data: np.ndarray of shape (n_samples, n_channels)
    """
    # Try whitespace separated; if it crashes, change 'delimiter' to ',' or '\t'
    data = np.loadtxt(path)
    return data


def bandpower_psd(signal_1d: np.ndarray, fs: int = FS) -> dict:
    """
    Compute bandpower for a 1D signal using Welch's PSD.
    Returns dict {band_name: power}.
    """
    f, psd = welch(signal_1d, fs=fs, nperseg=fs * 2)

    bp = {}
    for band_name, (fmin, fmax) in BANDS.items():
        idx = np.logical_and(f >= fmin, f <= fmax)
        bp[band_name] = np.trapz(psd[idx], f[idx])  # integral in that band
    return bp


def extract_features_from_trial(eeg: np.ndarray) -> np.ndarray:
    """
    Compute bandpower features for all channels of one trial.

    Args:
        eeg: np.ndarray, shape (n_samples, n_channels)

    Returns:
        feat: np.ndarray, shape (n_channels * n_bands,)
    """
    n_samples, n_channels = eeg.shape
    all_feats = []

    for ch in range(n_channels):
        sig = eeg[:, ch]
        bp = bandpower_psd(sig, fs=FS)
        # order of bands is fixed by BANDS.keys()
        all_feats.extend([bp[bname] for bname in BANDS.keys()])

    return np.array(all_feats, dtype=np.float32)


def collect_stew_trials(stew_root: str):
    """
    Scan STEW folder, find subXX_hi + subXX_lo files,
    extract bandpower features, and return X, y.

    Label convention:
        hi  -> 1  (high workload)
        lo  -> 0  (low workload)
    """
    X = []
    y = []
    meta = []   # keep file info for debugging / analysis

    # hi files
    hi_files = sorted(glob.glob(os.path.join(stew_root, "sub*_hi*")))
    lo_files = sorted(glob.glob(os.path.join(stew_root, "sub*_lo*")))

    print(f"Found {len(hi_files)} high-load files and {len(lo_files)} low-load files.")

    for label, file_list in [(1, hi_files), (0, lo_files)]:
        for fpath in file_list:
            try:
                eeg = load_eeg_file(fpath)
                # ensure 2D (n_samples, n_channels)
                if eeg.ndim == 1:
                    # single channel, reshape
                    eeg = eeg[:, None]

                feats = extract_features_from_trial(eeg)
                X.append(feats)
                y.append(label)
                meta.append(os.path.basename(fpath))
            except Exception as e:
                print(f"[WARN] Failed to process {fpath}: {e}")

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    meta = np.array(meta)

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    return X, y, meta


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def run_stew_pipeline():
    # 1. Load & featurize STEW
    X, y, meta = collect_stew_trials(STEW_ROOT)

    # 2. Train/val split (subject-independent split would be better; this is simple trial split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 3. Simple classifier: StandardScaler + Logistic Regression
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, n_jobs=-1)
    )

    clf.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Test Accuracy ===")
    print(f"{acc * 100:.2f}%")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["low", "high"]))

    # 5. Save model
    out_path = os.path.join(STEW_ROOT, "stew_logreg_bandpower.joblib")
    dump(clf, out_path)
    print(f"\nModel saved to: {out_path}")


if __name__ == "__main__":
    run_stew_pipeline()
