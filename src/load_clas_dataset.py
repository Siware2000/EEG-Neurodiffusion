import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ----- UPDATE THIS ROOT PATH IF NEEDED -----
CLAS_ROOT = r"D:\EEG-Neurodiffusion\Data Set\CLAS_Database\CLAS_Database\CLAS"


def _read_table(path: str) -> pd.DataFrame:
    """
    Helper that tries CSV first, then Excel.
    Works for files like:
      Part1_Block_Details.csv / .xlsx
      1_ecg_mathtest.csv / .xlsx
    """
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path)


def list_participants() -> List[str]:
    parts_dir = os.path.join(CLAS_ROOT, "Participants")
    parts = sorted(
        d for d in os.listdir(parts_dir)
        if d.lower().startswith("part") and os.path.isdir(os.path.join(parts_dir, d))
    )
    return parts


def load_block_details(part_name: str) -> pd.DataFrame:
    """
    Loads the Block_details file for one participant.
    Example filenames:
      Part1_Block_Details.csv / .xlsx
    """
    bd_dir = os.path.join(CLAS_ROOT, "Block_details")
    pattern = os.path.join(bd_dir, f"{part_name}_Block_Details*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No Block_details file found for {part_name}")
    df = _read_table(matches[0])

    # Show columns once to help you know where cognitive load label is
    print(f"\n[{part_name}] Block_details head:")
    print(df.head())
    print("Columns:", list(df.columns))
    return df


def load_signals_for_block(part_name: str, block_prefix: str) -> pd.DataFrame:
    """
    Loads ECG and GSR/PPG signals for a given block of one participant.

    In by_block folder filenames look like:
      2_ecg_mathtest.csv, 2_gsr_ppg_mathtest.csv, etc.

    block_prefix = '2'  (for block 2)
    """
    by_block_dir = os.path.join(CLAS_ROOT, "Participants", part_name, "by_block")

    # ECG file
    ecg_pattern = os.path.join(by_block_dir, f"{block_prefix}_ecg*")
    ecg_files = glob.glob(ecg_pattern)
    if not ecg_files:
        raise FileNotFoundError(f"No ECG file for {part_name}, block {block_prefix}")
    ecg_df = _read_table(ecg_files[0])

    # GSR/PPG file (optional)
    gsr_pattern = os.path.join(by_block_dir, f"{block_prefix}_gsr_ppg*")
    gsr_files = glob.glob(gsr_pattern)
    if gsr_files:
        gsr_df = _read_table(gsr_files[0])
        # Simple concat along columns (align by row index)
        sig_df = pd.concat([ecg_df, gsr_df], axis=1)
    else:
        sig_df = ecg_df

    return sig_df


def build_clas_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function:
      - iterates over all participants and blocks
      - loads signals
      - attaches a 'cognitive load' label from Block_details
      - returns X (n_samples, n_timesteps, n_channels) and y (n_samples)
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    parts = list_participants()
    print("Found participants:", parts)

    for part_name in parts:
        print(f"\n=== Processing {part_name} ===")

        # Skip duplicate / copy folders explicitly (like 'Part25 - Copy')
        if "copy" in part_name.lower():
            print(f"[INFO] Skipping duplicate folder: {part_name}")
            continue

        # Load block meta for this participant
        try:
            block_df = load_block_details(part_name)
        except FileNotFoundError as e:
            print(f"[WARN] {e} – skipping this participant.")
            continue

        # ---- IMPORTANT: choose the column that represents load level ----
        # Open the head printout and set the correct column below.
        # Examples (you will see the exact names in your printout):
        #   'CognitiveLoad', 'CL', 'Difficulty', etc.
        #
        # For now we assume there is a column named 'CL'.
        # -------------------------------------------------
        # Ensure we have a cognitive-load label column "CL"
        # -------------------------------------------------
        if "CL" not in block_df.columns:
            # Try to find the 'Block Type' column
            bt_col = None
            for c in block_df.columns:
                if "Block Type" in c:
                    bt_col = c
                    break

            if bt_col is None:
                print(f"[WARN] No 'CL' or 'Block Type' column for {part_name}, skipping this participant.")
                continue

            print(f"[WARN] Column 'CL' not found for {part_name}, "
                  f"inferring labels from '{bt_col}' (Baseline/Neutral/Pictures/Video clip).")

            def map_block_type_to_cl(bt):
                s = str(bt).strip().lower()
                if "baseline" in s:
                    return 0  # low load
                if "neutral" in s:
                    return 1  # medium load
                # Pictures, Video clip, and any other stimulus -> high load
                return 2  # high load

            block_df["CL"] = block_df[bt_col].apply(map_block_type_to_cl)

        for idx, row in block_df.iterrows():
            # Often there's a block id column like 'Block', 'BlockNo', etc.
            # Here we try 'Block' first, otherwise use the row index + 1.
            if "Block" in block_df.columns:
                block_id = row["Block"]
            else:
                block_id = idx + 1  # fallback

            block_prefix = str(int(block_id))

            try:
                sig_df = load_signals_for_block(part_name, block_prefix)
            except FileNotFoundError as e:
                print("  ", e)
                continue

            # Convert signal to numpy [T, C]
            sig_arr = sig_df.to_numpy(dtype="float32")
            X_list.append(sig_arr)

            # Map label from 'CL' column (you can map to 0/1/2 later)
            y_list.append(int(row["CL"]))

    if not X_list:
        raise RuntimeError("No samples loaded. Check file patterns and column names.")

    # To make shapes consistent, pad / cut each block to same length
    max_len = max(x.shape[0] for x in X_list)
    n_channels = X_list[0].shape[1]

    X = np.zeros((len(X_list), max_len, n_channels), dtype="float32")
    for i, arr in enumerate(X_list):
        L = min(max_len, arr.shape[0])
        X[i, :L, :] = arr[:L, :]

    y = np.array(y_list, dtype="int64")

    print("\nFinal dataset shape:")
    print("  X:", X.shape)  # (n_blocks, T, C)
    print("  y:", y.shape, "unique labels:", np.unique(y))

    return X, y


if __name__ == "__main__":
    # Quick test
    X, y = build_clas_dataset()
