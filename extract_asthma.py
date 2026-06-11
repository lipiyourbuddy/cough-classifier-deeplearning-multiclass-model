import pandas as pd
import shutil, os

# ── Config ─────────────────────────────────────────────────────
COSWARA_ROOT = r"E:\Coswara-Data-master"
CSV_PATH     = os.path.join(COSWARA_ROOT, "combined_data.csv")
OUTPUT_DIR   = r"C:\Users\Admin\Python workspace\Major Project\midsem - 5th draft\CovidCoughNet\Data\FINAL_AUDIO\asthma"
COUGH_FILES  = ["cough-heavy.wav", "cough-shallow.wav"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load & filter ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, sep=',')
df['asthma'] = df['asthma'].astype(str).str.strip().str.upper()

asthma_df = df[
    (df['asthma'] == 'TRUE') &
    (~df['covid_status'].str.contains('positive', case=False, na=False))
]
print(f"Attempting to copy audio for {len(asthma_df)} participants...")

# ── Copy audio ─────────────────────────────────────────────────
copied, not_found = 0, []

for _, row in asthma_df.iterrows():
    uid = str(row['id']).strip()
    found = False

    for entry in os.scandir(COSWARA_ROOT):
        if not entry.is_dir():
            continue
        participant_path = os.path.join(entry.path, uid)
        if not os.path.isdir(participant_path):
            continue

        for cough_file in COUGH_FILES:
            src = os.path.join(participant_path, cough_file)
            if os.path.exists(src):
                dst = os.path.join(OUTPUT_DIR, f"{uid}_{cough_file}")
                shutil.copy2(src, dst)
                copied += 1
                found = True

    if not found:
        not_found.append(uid)

# ── Summary ────────────────────────────────────────────────────
print(f"✅ Copied  : {copied} files")
print(f"⚠️  Missing : {len(not_found)} participants had no audio")
if not_found:
    print("Missing IDs:", not_found[:10], "..." if len(not_found) > 10 else "")