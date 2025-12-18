from pathlib import Path

ROOT = Path("D:/EEG-Neurodiffusion/topomaps_augmented/all")

print("\n=== AUGMENTED DATASET SUMMARY ===")
total = 0
for c in sorted(ROOT.iterdir()):
    n = len(list(c.glob("*.png")))
    total += n
    print(f"Class {c.name}: {n} images")

print(f"TOTAL images: {total}")
print("================================")
