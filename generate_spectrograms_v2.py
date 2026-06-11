# generate_spectrograms_v2.py
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_DIR  = "Data/processed_audio"
OUTPUT_DIR = "Data/spectrograms_v2"
CLASSES    = ["covid", "healthy", "asthma", "other_resp"]
TARGET_SR  = 16000
N_MELS     = 128
N_FFT      = 1024
HOP_LEN    = 512
F_MIN      = 50
F_MAX      = 8000
N_MFCC     = 128

MIN_RMS         = 0.001
MAX_RMS         = 0.999
MIN_ACTIVE_FRAC = 0.2

for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def is_valid_audio(y):
    rms = np.sqrt(np.mean(y**2))
    if rms < MIN_RMS or rms > MAX_RMS:
        return False
    frame_size = len(y) // 10
    active = sum(
        1 for i in range(10)
        if np.sqrt(np.mean(y[i*frame_size:(i+1)*frame_size]**2)) > MIN_RMS
    )
    return (active / 10) >= MIN_ACTIVE_FRAC

def normalize_channel(data):
    mn, mx = data.min(), data.max()
    if mx - mn < 1e-6:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)

def make_3channel(y, sr):
    # Channel 1: Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LEN, fmin=F_MIN, fmax=F_MAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Channel 2: MFCC (vocal tract shape)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LEN, fmin=F_MIN, fmax=F_MAX
    )

    # Channel 3: Spectral contrast (distinguishes harmonic vs noise)
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, fmin=F_MIN
    )
    # resize contrast to match mel shape (128 bins)
    contrast_resized = np.resize(contrast, (N_MELS, mel_db.shape[1]))

    # Normalize each channel to [0, 1]
    ch1 = normalize_channel(mel_db)
    ch2 = normalize_channel(mfcc)
    ch3 = normalize_channel(contrast_resized)

    # Stack as (H, W, 3)
    return np.stack([ch1, ch2, ch3], axis=-1)

def save_spectrogram(audio_path, out_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)

    if not is_valid_audio(y):
        return False

    img = make_3channel(y, sr)

    if img[:,:,0].max() - img[:,:,0].min() < 0.01:
        return False

    # Save as PNG (3 channel)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    plt.imshow(img, aspect='auto', origin='lower')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    return True

total_saved, total_rejected = {}, {}

for cls in CLASSES:
    src_dir = os.path.join(INPUT_DIR, cls)
    out_dir = os.path.join(OUTPUT_DIR, cls)
    files   = [f for f in os.listdir(src_dir) if f.endswith(".wav")]
    saved, rejected = 0, 0

    print(f"\nGenerating {cls} ({len(files)} files)...")

    for fname in files:
        try:
            src      = os.path.join(src_dir, fname)
            out_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(out_dir, out_name)
            if save_spectrogram(src, out_path):
                saved += 1
            else:
                rejected += 1
        except Exception as e:
            print(f"  ❌ {fname}: {e}")
            rejected += 1

    total_saved[cls]    = saved
    total_rejected[cls] = rejected
    print(f"  ✅ Saved: {saved}  🗑 Rejected: {rejected}")

print("\n── Final counts ──────────────────────────")
for cls in CLASSES:
    print(f"  {cls:12s}: {total_saved[cls]} spectrograms")
print(f"\n  Total saved   : {sum(total_saved.values())}")
print(f"  Total rejected: {sum(total_rejected.values())}")