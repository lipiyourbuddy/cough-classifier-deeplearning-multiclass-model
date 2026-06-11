import os
import librosa
import soundfile as sf
import numpy as np

BASE_DIR     = "Data/FINAL_AUDIO"
TARGET_SR    = 16000
TARGET_FILES = 600   # target raw files per class before chunking
                     # ~600 raw files × ~1.5 chunks each ≈ 900 chunks

CLASSES_TO_AUGMENT = ["asthma", "other_resp"]  # covid/healthy already sufficient

def add_noise(y, factor=0.005):
    return y + factor * np.random.randn(len(y))

def time_stretch(y, rate=None):
    rate = rate or np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, steps=None):
    steps = steps or np.random.uniform(-3, 3)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_shift(y, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def random_gain(y, min_gain=0.7, max_gain=1.3):
    gain = np.random.uniform(min_gain, max_gain)
    return np.clip(y * gain, -1.0, 1.0)

AUGMENTATIONS = [add_noise, time_stretch, pitch_shift, time_shift, random_gain]

for cls in CLASSES_TO_AUGMENT:
    input_dir = os.path.join(BASE_DIR, cls)
    files     = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    existing  = len(files)
    needed    = TARGET_FILES - existing

    if needed <= 0:
        print(f"\n{cls}: already has {existing} raw files, skipping")
        continue

    print(f"\n{cls}: {existing} raw files → generating {needed} augmented raw files...")

    generated, aug_idx = 0, 0

    while generated < needed:
        fname    = files[generated % len(files)]
        src_path = os.path.join(input_dir, fname)
        base     = os.path.splitext(fname)[0]

        try:
            y, sr  = librosa.load(src_path, sr=TARGET_SR)
            aug_fn = AUGMENTATIONS[aug_idx % len(AUGMENTATIONS)]

            if aug_fn == pitch_shift:
                y_aug = aug_fn(y, sr)
            else:
                y_aug = aug_fn(y)

            # normalize but preserve natural length — do NOT fix length here
            # chunking will handle length later
            y_aug = librosa.util.normalize(y_aug)

            out_name = f"{base}_aug{aug_idx:04d}.wav"
            sf.write(os.path.join(input_dir, out_name), y_aug, TARGET_SR)

            generated += 1
            aug_idx   += 1

        except Exception as e:
            print(f"  ❌ {fname}: {e}")
            aug_idx += 1
            continue

    print(f"  ✅ {cls} now has {existing + generated} raw files")

print("\n✅ Raw augmentation complete")
print("\n── Final raw file counts ─────────────────")
for cls in ["covid", "healthy", "asthma", "other_resp"]:
    d = os.path.join(BASE_DIR, cls)
    if os.path.exists(d):
        print(f"  {cls:12s}: {len([f for f in os.listdir(d) if f.endswith('.wav')])} files")