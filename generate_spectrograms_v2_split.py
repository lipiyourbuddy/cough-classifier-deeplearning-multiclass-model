# split_by_subject_v2.py
import os, re, shutil, numpy as np
from collections import defaultdict

SPECTROGRAM_DIR = "Data/spectrograms_v2"
OUTPUT_DIR      = "Data/spectrograms_v2_split"
CLASSES         = ["covid", "healthy", "asthma", "other_resp"]
VAL_SPLIT       = 0.2
SEED            = 42
np.random.seed(SEED)

def extract_subject_id(filename, cls):
    name = os.path.splitext(filename)[0]
    if cls in ["covid", "healthy"]:
        match = re.match(r"coswara_(.+?)(\d+)$", name)
        return match.group(1) if match else name
    else:
        return name.split("_")[0]

for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in CLASSES:
    src_dir = os.path.join(SPECTROGRAM_DIR, cls)
    files   = [f for f in os.listdir(src_dir) if f.endswith(".png")]

    subject_to_files = defaultdict(list)
    for f in files:
        subject_to_files[extract_subject_id(f, cls)].append(f)

    subjects = list(subject_to_files.keys())
    np.random.shuffle(subjects)
    n_val       = int(len(subjects) * VAL_SPLIT)
    val_subjs   = set(subjects[:n_val])

    train_count = val_count = 0
    for sid, flist in subject_to_files.items():
        split_name = "val" if sid in val_subjs else "train"
        for fname in flist:
            shutil.copy2(
                os.path.join(src_dir, fname),
                os.path.join(OUTPUT_DIR, split_name, cls, fname)
            )
            if split_name == "train": train_count += 1
            else: val_count += 1

    print(f"{cls}: {len(subjects)-n_val} train subjects ({train_count} files) | "
          f"{n_val} val subjects ({val_count} files)")

print("\n✅ Split complete →", OUTPUT_DIR)