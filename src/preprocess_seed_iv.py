# src/preprocess_seed_iv.py

import scipy.io as sio
import numpy as np
import mne
import os

SEED_IV_PATH = r"Dpower:\EEG-Neurodiffusion\Data Set\SEED_IV\seed_iv\eeg_raw_data"

def load_seed_iv_trial(mat_path):
    """Loads one SEED-IV .mat file and returns 24 trials."""
    mat = sio.loadmat(mat_path)

    # find EEG* keys (e.g., wll_eeg1, wll_eeg2…)
    eeg_keys = [k for k in mat.keys() if "eeg" in k.lower()]
    trials = []

    for k in eeg_keys:
        arr = mat[k]
        # arr shape: (62, time)
        if arr.ndim == 2:
            trials.append(arr)

    return trials

def preprocess_trial(raw_trial, sfreq=200):
    """
    Convert raw trial → MNE object → apply filtering → return filtered array.
    """
    n_channels, n_times = raw_trial.shape
    info = mne.create_info(
        ch_names=[f"EEG{i+1}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg"
    )

    raw = mne.io.RawArray(raw_trial, info)

    # bandpass 1–45 Hz
    raw.filter(1., 45., fir_design='firwin')

    # common average reference
    raw.set_eeg_reference('average')

    return raw.get_data()

def load_and_preprocess_seed_iv():
    X = []
    y = []

    print("=== Loading SEED-IV ===")
    for session in ["1", "2", "3"]:
        folder = os.path.join(SEED_IV_PATH, session)
        files = [f for f in os.listdir(folder) if f.endswith(".mat")]

        for f in files:
            fpath = os.path.join(folder, f)
            try:
                trials = load_seed_iv_trial(fpath)
                # simple label: subject ID mod 4  →  0,1,2,3
                label = int(f.split("_")[0]) % 4
            except Exception as e:
                print(f"[WARN] Skipping corrupted file: {f} ({e})")
                continue

            for t in trials:
                filtered = preprocess_trial(t)   # shape: (62, T_i)
                X.append(filtered)
                y.append(label)

    # ---------- NEW PART: enforce fixed length ----------
    # 1) compute all time lengths
    lengths = [arr.shape[1] for arr in X]
    min_len = min(lengths)

    print(f"Found {len(X)} trials.")
    print(f"Time lengths range: min={min_len}, max={max(lengths)}")
    print(f"Cropping all trials to common length: {min_len} samples")

    # 2) crop each trial to min_len and stack
    X = np.stack([arr[:, :min_len] for arr in X], axis=0).astype("float32")
    y = np.array(y, dtype="int64")

    print("Final shape:", X.shape, y.shape)   # (N_trials, 62, min_len)
    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess_seed_iv()
