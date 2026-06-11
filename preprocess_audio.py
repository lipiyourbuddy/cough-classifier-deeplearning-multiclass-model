import os
import librosa
import soundfile as sf
import numpy as np

INPUT_DIR  = "Data/FINAL_AUDIO"
OUTPUT_DIR = "Data/processed_audio"

TARGET_SR       = 16000
TARGET_DURATION = 5
TARGET_LENGTH   = TARGET_SR * TARGET_DURATION
MIN_LENGTH      = TARGET_SR * 1
MIN_RAW_DURATION = 0.5  

CLASSES = ["covid", "healthy", "asthma", "other_resp"]

for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

skipped   = 0
processed = 0

def process_file(path, label):
    global skipped, processed
    try:
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

       
        if len(y) < TARGET_SR * MIN_RAW_DURATION:
            skipped += 1
            return

        y, _ = librosa.effects.trim(y, top_db=20)

        if len(y) == 0:
            skipped += 1
            return

        y = librosa.util.normalize(y)

        start     = 0
        chunk_idx = 0
        base_name = os.path.splitext(os.path.basename(path))[0]

        while start < len(y):
            chunk = y[start : start + TARGET_LENGTH]

            if len(chunk) < MIN_LENGTH:
                break

            if len(chunk) < TARGET_LENGTH:
                chunk = np.pad(chunk, (0, TARGET_LENGTH - len(chunk)))

            save_path = os.path.join(OUTPUT_DIR, label, f"{base_name}_chunk{chunk_idx:03d}.wav")
            sf.write(save_path, chunk, TARGET_SR)

            chunk_idx += 1
            start     += TARGET_LENGTH
            processed += 1

    except Exception as e:
        skipped += 1

for label in CLASSES:
    folder = os.path.join(INPUT_DIR, label)
    if not os.path.exists(folder):
        print(f" Skipping {label} — folder not found")
        continue

    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"\nProcessing {label} ({len(files)} files)...")

    for file in files:
        process_file(os.path.join(folder, file), label)

print("\n Audio preprocessing complete")
print(f"   Chunks created : {processed}")
print(f"   Skipped        : {skipped}")