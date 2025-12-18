import os
from typing import List, Tuple

import numpy as np
import scipy.io as sio


# === UPDATE THIS PATH IF NEEDED ===
# If your folder is F:\EEG-Neurodiffusion\Data Set\SEED_IV\seed_iv\eeg_raw_data
SEED_IV_ROOT = r"D:\EEG-Neurodiffusion\Data Set\SEED_IV\seed_iv\eeg_raw_data"
# If instead it is F:\EEG-Neurodiffusion\Data Set\SEED_IV\eeg_raw_data
# then change to:
# SEED_IV_ROOT = r"F:\EEG-Neurodiffusion\Data Set\SEED_IV\eeg_raw_data"


def load_seed_iv_raw() -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load SEED-IV raw EEG from all 3 sessions.

    Your .mat files have keys like:
      'cz_eeg1', 'cz_eeg2', ..., 'cz_eeg24'
      'tyc_eeg1', ..., 'tyc_eeg24', etc.

    We will:
      - read all keys that contain 'eeg'
      - treat each as one trial (channels x time)
      - assign placeholder labels 0..3 (4 emotions) in a cyclic way
        (you can replace later with the official label list from SEED-IV)
    """

    sessions = ["1", "2", "3"]
    X: List[np.ndarray] = []
    y_list: List[int] = []

    trial_counter = 0

    for ses in sessions:
        ses_dir = os.path.join(SEED_IV_ROOT, ses)
        if not os.path.isdir(ses_dir):
            print(f"[WARN] Session folder not found: {ses_dir}")
            continue

        print(f"\n=== Loading session {ses} from {ses_dir} ===")

        mat_files = sorted(
            f for f in os.listdir(ses_dir)
            if f.lower().endswith(".mat")
        )

        if not mat_files:
            print(f"[WARN] No .mat files found in {ses_dir}")
            continue

        for fname in mat_files:
            fpath = os.path.join(ses_dir, fname)
            print(f"  Loading {fname} ...", end=" ")

            # some files may be corrupted or in unsupported format
            try:
                mat = sio.loadmat(fpath, squeeze_me=True)
            except Exception as e:
                print(f"FAILED to read file (skipping). Reason: {e}")
                continue

            # pick all variables that look like EEG trials
            eeg_keys = [
                k for k in mat.keys()
                if not k.startswith("__") and "eeg" in k.lower()
            ]

            if not eeg_keys:
                print("FAILED (no '*eeg*' variables). Keys:", list(mat.keys()))
                continue

            eeg_keys = sorted(eeg_keys)

            n_added = 0
            for k in eeg_keys:
                eeg = np.asarray(mat[k], dtype=np.float32)

                # We expect 2D array: [channels, time] or [time, channels]
                if eeg.ndim != 2:
                    # if 3D etc., you can customize here; for now skip
                    print(f"\n    [WARN] Variable {k} has shape {eeg.shape}, not 2D. Skipping this variable.")
                    continue

                # If time dimension is smaller than channels, you may transpose.
                # For now, we leave it as-is; you can inspect shapes later.
                X.append(eeg)
                label = trial_counter % 4   # placeholder: 4 emotions (0..3)
                y_list.append(label)
                trial_counter += 1
                n_added += 1

            print(f"OK, added {n_added} trials from {fname}.")

    if not X:
        raise RuntimeError("No EEG trials loaded – check SEED_IV_ROOT path or file structure.")

    y = np.array(y_list, dtype=np.int64)

    print("\n=== Summary ===")
    print("Total trials:", len(X))
    print("Label shape:", y.shape)
    print("Unique labels:", np.unique(y))

    return X, y



if __name__ == "__main__":
    X, y = load_seed_iv_raw()

    # Inspect first trial
    first = X[0]
    print("\nFirst trial shape:", first.shape)   # (n_channels, n_samples)
    print("First label:", y[0])
