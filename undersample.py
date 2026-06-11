import os
import shutil
import random

BASE_DIR    = "Data/processed_audio"
OUTPUT_DIR  = "Data/processed_audio_balanced"
TARGET      = 648  # match the smallest class
CLASSES     = ["covid", "healthy", "asthma", "other_resp"]
SEED        = 42

random.seed(SEED)

for cls in CLASSES:
    src = os.path.join(BASE_DIR, cls)
    dst = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.endswith(".wav")]

    # undersample if more than target, keep all if less
    selected = random.sample(files, min(TARGET, len(files)))

    for f in selected:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    print(f"{cls:12s}: {len(selected)} chunks copied")

print("\n✅ Balanced dataset ready in Data/processed_audio_balanced")