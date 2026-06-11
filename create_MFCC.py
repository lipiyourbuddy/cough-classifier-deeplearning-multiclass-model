import os
import csv
import numpy as np
import librosa

# Paths
audio_folder = "Data/COUGHVID_audio_pitch_shift_4/"
output_csv = "Data/COUGHVID_features.csv"

# CSV header
header = [
    "filename",
    "chroma_stft",
    "rmse",
    "spectral_centroid",
    "spectral_bandwidth",
    "rolloff",
    "zero_crossing_rate"
]

# Add 20 MFCC columns
for i in range(1, 21):
    header.append(f"mfcc{i}")

header.append("label")

# Create CSV file
with open(output_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)

# Loop through audio files
for file in os.listdir(audio_folder):

    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(audio_folder, file)

    print("Processing:", file)

    # Determine label from filename
    if "covid" in file.lower():
        label = 1
    else:
        label = 0

    # Load audio
    y, sr = librosa.load(file_path, mono=True, duration=5)

    # Feature extraction
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Combine features
    row = [
        file,
        np.mean(chroma_stft),
        np.mean(rmse),
        np.mean(spec_cent),
        np.mean(spec_bw),
        np.mean(rolloff),
        np.mean(zcr)
    ]

    for m in mfcc:
        row.append(np.mean(m))

    row.append(label)

    # Write to CSV
    with open(output_csv, "a", newline="") as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(row)

print("Feature extraction complete.")
print("Dataset saved to:", output_csv)