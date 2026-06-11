import os
import shutil
import pandas as pd

# Coswara dataset location
coswara_root = r"E:\Coswara-Data-master"

# metadata file
metadata_file = os.path.join(coswara_root, "combined_data.csv")

# destination folders
covid_dest = r"C:\Users\Admin\Python workspace\Major Project\midsem - 2nd draft\CovidCoughNet\Data\Coswara_audio\covid"
healthy_dest = r"C:\Users\Admin\Python workspace\Major Project\midsem - 2nd draft\CovidCoughNet\Data\Coswara_audio\healthy"

os.makedirs(covid_dest, exist_ok=True)
os.makedirs(healthy_dest, exist_ok=True)

df = pd.read_csv(metadata_file)

copied_covid = 0
copied_healthy = 0
skipped = 0

for root, dirs, files in os.walk(coswara_root):

    if "cough-heavy.wav" in files:

        wav_path = os.path.join(root, "cough-heavy.wav")

        participant_id = os.path.basename(root)

        row = df[df["id"] == participant_id]

        if row.empty:
            continue

        status = str(row.iloc[0]["covid_status"]).lower()

        filename = participant_id + ".wav"

        if "positive" in status:

            dest = os.path.join(covid_dest, filename)

            if not os.path.exists(dest):
                shutil.copy(wav_path, dest)
                copied_covid += 1
            else:
                skipped += 1

        elif status == "healthy":

            dest = os.path.join(healthy_dest, filename)

            if not os.path.exists(dest):
                shutil.copy(wav_path, dest)
                copied_healthy += 1
            else:
                skipped += 1

print("Covid samples copied:", copied_covid)
print("Healthy samples copied:", copied_healthy)
print("Duplicates skipped:", skipped)