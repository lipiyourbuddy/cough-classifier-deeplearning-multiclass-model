import os
import librosa
import soundfile as sf
import numpy as np

BASE_DIR   = "Data/processed_audio"
TARGET_SR  = 16000
TARGET_LEN = 16000 * 5

CLASS_TARGETS = {
    "asthma"    : 900,
    "other_resp": 900,
}

def add_noise(y, factor=0.005):
    return y + factor * np.random.randn(len(y))

def time_stretch(y, rate=None):
    rate = rate or np.random.uniform(0.85, 1.15)
    y = librosa.effects.time_stretch(y, rate=rate)
    return librosa.util.fix_length(y, size=TARGET_LEN)

def pitch_shift(y, sr, steps=None):
    steps = steps or np.random.uniform(-3, 3)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_shift(y, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def random_gain(y, min_gain=0.7, max_gain=1.3):
    return np.clip(y * np.random.uniform(min_gain, max_gain), -1.0, 1.0)

AUGMENTATIONS = [add_noise, time_stretch, pitch_shift, time_shift, random_gain]

for cls, target in CLASS_TARGETS.items():
    input_dir = os.path.join(BASE_DIR, cls)

    all_files  = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    src_files  = [f for f in all_files if "caug" not in f]
    existing   = len(all_files)
    needed     = target - existing

    if needed <= 0:
        print(f"\n{cls}: already has {existing} chunks, skipping")
        continue

    synthetic_pct = (needed / target) * 100
    print(f"\n{cls}: {existing} chunks → generating {needed} more ({synthetic_pct:.0f}% chunk-level synthetic)")

    generated, aug_idx = 0, 0

    while generated < needed:
        fname    = src_files[aug_idx % len(src_files)]
        src_path = os.path.join(input_dir, fname)
        base     = os.path.splitext(fname)[0]

        try:
            y, sr = librosa.load(src_path, sr=TARGET_SR)

            if len(y) == 0:
                aug_idx += 1
                continue

            aug_fn = AUGMENTATIONS[aug_idx % len(AUGMENTATIONS)]
            y_aug  = aug_fn(y, sr) if aug_fn == pitch_shift else aug_fn(y)

            if len(y_aug) < TARGET_LEN:
                y_aug = np.pad(y_aug, (0, TARGET_LEN - len(y_aug)))
            else:
                y_aug = y_aug[:TARGET_LEN]

            y_aug = librosa.util.normalize(y_aug)

            out_name = f"{base}_caug{aug_idx:04d}.wav"
            sf.write(os.path.join(input_dir, out_name), y_aug, TARGET_SR)

            generated += 1
            aug_idx   += 1

        except Exception as e:
            print(f"  ❌ {fname}: {e}")
            aug_idx += 1
            continue

    print(f"   Done. {cls} now has {existing + generated} chunks")

print("\n Augmentation complete")
print("\n── Final chunk counts ────────────────────")
for cls in ["covid", "healthy", "asthma", "other_resp"]:
    d        = os.path.join(BASE_DIR, cls)
    total    = [f for f in os.listdir(d) if f.endswith(".wav")]
    caug     = [f for f in total if "caug" in f]
    raw_aug  = [f for f in total if "_aug" in f and "caug" not in f]
    orig     = [f for f in total if "_aug" not in f and "caug" not in f]
    print(f"  {cls:12s}: {len(total)} total | {len(orig)} original | {len(raw_aug)} raw-aug | {len(caug)} chunk-aug")