# src/load_and_preprocess.py

import os
from typing import Tuple

import mne
import numpy as np


def load_raw_auto(filepath: str) -> mne.io.BaseRaw:
    """
    Load EEG from common formats: .edf, .vhdr, .bdf, .fif
    (For .mat/.set, see comment below.)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".edf":
        raw = mne.io.read_raw_edf(filepath, preload=True)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(filepath, preload=True)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(filepath, preload=True)
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(filepath, preload=True)
    else:
        raise ValueError(
            f"Unsupported extension {ext}. "
            f"Add custom loader for .mat/.set/.csv in load_raw_auto()."
        )

    raw.rename_channels(lambda ch: ch.strip())
    return raw


def basic_preprocess(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    epoch_length: float = 2.0,
) -> mne.Epochs:
    """
    Simple preprocessing:
    - Bandpass filter 1–45 Hz
    - Remove DC
    - Create fixed-length epochs (no event labels yet)
    """
    raw = raw.copy().load_data()
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.set_eeg_reference("average")

    # Create dummy events every `epoch_length` seconds
    events = mne.make_fixed_length_events(raw, duration=epoch_length)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_length,
        baseline=None,
        preload=True,
    )
    return epochs


def debug_info(epochs: mne.Epochs) -> None:
    print("Epochs info:")
    print("  n_epochs:", len(epochs))
    print("  n_channels:", len(epochs.ch_names))
    print("  sfreq:", epochs.info["sfreq"])
    print("  duration (s):", epochs.tmax - epochs.tmin)
