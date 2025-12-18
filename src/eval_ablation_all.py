import argparse
from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

from dataset_topomap import TopomapImageDataset
from eval_vit_checkpoint import build_model, detect_backend, _unwrap_state


ABLATION_CHECKPOINTS = {
    "Real only": "vit_real_only.pt",
    "Real + Diffusion (raw)": "vit_raw_diffusion.pt",
    "Real + Selective Diffusion": "vit_selective_diffusion.pt",
    "Real + J2-Filtered Diffusion (Ours)": "vit_j2_ours.pt",
}


def evaluate_one(ckpt_path, dataset, device, seed=42):
    num_classes = len(dataset.class_to_idx)

    # same split for all models
    val_len = int(0.2 * len(dataset))
    train_len = len(dataset) - val_len
    _, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    state_raw = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_state(state_raw)
    backend = detect_backend(state)

    model = build_model(num_classes, backend).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1, backend


def main():
    parser = argparse.ArgumentParser("Freeze ablation table (journal-ready)")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out-dir", default="eval_ablation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = TopomapImageDataset(args.data_root, img_size=224)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n=== FINAL ABLATION RESULTS ===")
    for method, ckpt_name in ABLATION_CHECKPOINTS.items():
        ckpt_path = Path(args.ckpt_dir) / ckpt_name
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

        acc, f1, backend = evaluate_one(ckpt_path, dataset, device)

        print(f"{method:35s} | Acc = {acc:.4f} | F1 = {f1:.4f}")

        rows.append({
            "Method": method,
            "Accuracy": round(acc, 4),
            "Macro-F1": round(f1, 4),
            "Checkpoint": ckpt_name,
            "Backend": backend
        })

    # Save CSV (journal table)
    csv_path = out_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save TXT (human-readable)
    txt_path = out_dir / "ablation_results.txt"
    with open(txt_path, "w") as f:
        for r in rows:
            f.write(f"{r['Method']}: Acc={r['Accuracy']} | F1={r['Macro-F1']} | {r['Checkpoint']}\n")

    print(f"\n[DONE] Frozen ablation saved to:\n  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
