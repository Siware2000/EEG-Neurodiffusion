# src/generate_topomaps.py

import os
from typing import Optional

import numpy as np
import mne
import matplotlib.pyplot as plt
import cv2


def epoch_to_topomap_png(
    epochs: mne.Epochs,
    epoch_index: int,
    out_path: str,
    fmin: float = 8.0,
    fmax: float = 30.0,
    size: int = 128,
    cmap: str = "viridis",
) -> None:
    """
    Compute band-limited PSD (e.g. alpha+beta) and plot as scalp topomap.
    Saves it as a PNG image.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # (n_channels, n_times)
    epoch_data = epochs.get_data()[epoch_index]

    psd, freqs = mne.time_frequency.psd_array_welch(
        epoch_data,
        sfreq=epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        n_fft=256,
    )
    psd_mean = psd.mean(axis=1)

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(
        psd_mean,
        epochs.info,
        axes=ax,
        show=False,
        cmap=cmap,
        contours=0,
    )
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def generate_topomaps_for_epochs(
    epochs: mne.Epochs,
    out_dir: str,
    label: Optional[int] = None,
) -> None:
    """
    Save all epochs as images. Folder layout:
    out_dir/
        class0/
        class1/
        ...
    For now, if label is None → saves under 'unknown/'.
    """
    label_str = "unknown" if label is None else f"class{label}"
    base_dir = os.path.join(out_dir, label_str)
    os.makedirs(base_dir, exist_ok=True)

    for idx in range(len(epochs)):
        out_path = os.path.join(base_dir, f"epoch_{idx:04d}.png")
        epoch_to_topomap_png(epochs, idx, out_path)
