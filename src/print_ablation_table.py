
from pathlib import Path
import csv
import json

ROOT = Path("eval_ablation_final")

rows = []

for exp in sorted(ROOT.iterdir()):
    # ✅ Only process experiment directories
    if not exp.is_dir():
        continue

    metrics_file = exp / "metrics.json"
    if not metrics_file.exists():
        print(f"[WARN] Skipping {exp.name}: metrics.json not found")
        continue

    with open(metrics_file) as f:
        m = json.load(f)

    rows.append([
        exp.name,
        m["accuracy"],
        m["macro_f1"],
        Path(m["checkpoint"]).name
    ])

# =========================
# Save CSV
# =========================
csv_path = ROOT / "ablation_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Accuracy", "Macro-F1", "Checkpoint"])
    writer.writerows(rows)

# =========================
# Save TXT
# =========================
txt_path = ROOT / "ablation_results.txt"
with open(txt_path, "w") as f:
    for r in rows:
        f.write(f"{r[0]:<20}  Acc={r[1]:.4f}  Macro-F1={r[2]:.4f}\n")

# =========================
# Print to console
# =========================
print("\n=== FINAL ABLATION TABLE (FROZEN) ===")
for r in rows:
    print(f"{r[0]:<20}  Acc={r[1]:.4f}  Macro-F1={r[2]:.4f}")

print(f"\n[SAVED] {csv_path}")
print(f"[SAVED] {txt_path}")
