# split_by_subject.py
import os
import re
import shutil
import numpy as np
from collections import defaultdict

SPECTROGRAM_DIR = "Data/spectrograms"
OUTPUT_DIR      = "Data/spectrograms_split"
CLASSES         = ["covid", "healthy", "asthma", "other_resp"]
VAL_SPLIT       = 0.2
SEED            = 42
np.random.seed(SEED)

def extract_subject_id(filename, cls):
    name = os.path.splitext(filename)[0]  # remove .png

    if cls in ["covid", "healthy"]:
        # coswara_<subjectid><chunknum>  e.g. coswara_00xKcQMmcAhX8CODgBBLOe7Dm0T2
        # subject id is everything after 'coswara_' minus trailing digit(s)
        match = re.match(r"coswara_(.+?)(\d+)$", name)
        if match:
            return match.group(1)
        return name  # fallback

    else:
        # asthma/other_resp: <SUBJECTID>_cough-heavy_aug0000
        # subject id is the part before first underscore
        return name.split("_")[0]

for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in CLASSES:
    src_dir = os.path.join(SPECTROGRAM_DIR, cls)
    files   = [f for f in os.listdir(src_dir) if f.endswith(".png")]

    # Group files by subject
    subject_to_files = defaultdict(list)
    for f in files:
        sid = extract_subject_id(f, cls)
        subject_to_files[sid].append(f)

    subjects = list(subject_to_files.keys())
    np.random.shuffle(subjects)

    n_val      = int(len(subjects) * VAL_SPLIT)
    val_subjs  = set(subjects[:n_val])
    train_subjs = set(subjects[n_val:])

    train_count, val_count = 0, 0

    for sid, flist in subject_to_files.items():
        split_name = "val" if sid in val_subjs else "train"
        for fname in flist:
            src  = os.path.join(src_dir, fname)
            dst  = os.path.join(OUTPUT_DIR, split_name, cls, fname)
            shutil.copy2(src, dst)
            if split_name == "train":
                train_count += 1
            else:
                val_count += 1

    print(f"{cls}: {len(train_subjs)} train subjects ({train_count} files) | "
          f"{len(val_subjs)} val subjects ({val_count} files)")

print("\nSplit complete. Data saved to:", OUTPUT_DIR)