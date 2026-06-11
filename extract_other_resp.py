import pandas as pd
import shutil, os

COSWARA_ROOT = r"E:\Coswara-Data-master"
CSV_PATH     = os.path.join(COSWARA_ROOT, "combined_data.csv")
OUTPUT_DIR   = r"C:\Users\Admin\Python workspace\Major Project\midsem - 6th draft\CovidCoughNet\Data\FINAL_AUDIO\other_resp"
COUGH_FILES  = ["cough-heavy.wav", "cough-shallow.wav"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
for col in ['others_resp', 'bd', 'cld', 'asthma']:
    df[col] = df[col].astype(str).str.strip().str.upper()

resp_df = df[
    (
        (df['others_resp'] == 'TRUE') |
        (df['bd'] == 'TRUE')          |
        (df['cld'] == 'TRUE')
    ) &
    (df['asthma'] != 'TRUE') &
    (~df['covid_status'].str.contains('positive', case=False, na=False))
]

print(f"Extracting audio for {len(resp_df)} participants...")

copied, not_found = 0, []

for _, row in resp_df.iterrows():
    uid   = str(row['id']).strip()
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
                found  = True

    if not found:
        not_found.append(uid)

print(f"✅ Copied  : {copied} files")
print(f"⚠️  Missing : {len(not_found)} participants had no audio")