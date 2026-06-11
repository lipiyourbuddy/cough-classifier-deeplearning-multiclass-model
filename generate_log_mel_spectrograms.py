import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

INPUT_DIR = "Data/processed_audio"
OUTPUT_DIR = "Data/spectrograms"
CLASSES   = ["covid", "healthy", "asthma", "other_resp"]
TARGET_SR = 16000
N_MELS    = 128
N_FFT     = 1024
HOP_LEN   = 512
F_MIN     = 50
F_MAX     = 8000

# ── Quality thresholds ─────────────────────────────────────────
MIN_RMS         = 0.001   # below this = essentially silent
MAX_RMS         = 0.999   # above this = clipped/saturated
MIN_ACTIVE_FRAC = 0.2     # at least 40% of the chunk must be non-silent
                          # catches zero-padded second halves

for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def is_valid_audio(y):
    rms = np.sqrt(np.mean(y**2))
    if rms < MIN_RMS or rms > MAX_RMS:
        return False
    # check active fraction — split into 10 frames, count non-silent
    frame_size  = len(y) // 10
    active      = sum(
        1 for i in range(10)
        if np.sqrt(np.mean(y[i*frame_size:(i+1)*frame_size]**2)) > MIN_RMS
    )
    if active / 10 < MIN_ACTIVE_FRAC:
        return False
    return True

def save_logmel(audio_path, out_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)

    if not is_valid_audio(y):
        return False

    mel    = librosa.feature.melspectrogram(
                 y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
                 hop_length=HOP_LEN, fmin=F_MIN, fmax=F_MAX
             )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # check spectrogram isn't flat (pure black/orange)
    if mel_db.max() - mel_db.min() < 10:
        return False

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.set_axis_off()
    librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LEN,
        fmin=F_MIN, fmax=F_MAX,
        cmap='inferno', ax=ax
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return True

total_saved    = {}
total_rejected = {}

for cls in CLASSES:
    src_dir  = os.path.join(INPUT_DIR, cls)
    out_dir  = os.path.join(OUTPUT_DIR, cls)
    files    = [f for f in os.listdir(src_dir) if f.endswith(".wav")]
    saved, rejected = 0, 0

    print(f"\nGenerating {cls} spectrograms ({len(files)} files)...")

    for fname in files:
        try:
            src      = os.path.join(src_dir, fname)
            out_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(out_dir, out_name)

            if save_logmel(src, out_path):
                saved += 1
            else:
                rejected += 1

        except Exception as e:
            print(f"  ❌ {fname}: {e}")
            rejected += 1

    total_saved[cls]    = saved
    total_rejected[cls] = rejected
    print(f"  ✅ Saved   : {saved}")
    print(f"  🗑  Rejected: {rejected}")

print("\n── Final counts ──────────────────────────")
for cls in CLASSES:
    print(f"  {cls:10s}: {total_saved[cls]} spectrograms")
print(f"\n  Total saved   : {sum(total_saved.values())}")
print(f"  Total rejected: {sum(total_rejected.values())}")