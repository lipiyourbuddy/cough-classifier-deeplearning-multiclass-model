import os
import pandas as pd

data = []

covid_path = "Data/COUGHVID_audio/covid"
healthy_path = "Data/COUGHVID_audio/healthy"

# covid = 1
for file in os.listdir(covid_path):
    if file.endswith(".wav"):
        data.append([file,1])

# healthy = 0
for file in os.listdir(healthy_path):
    if file.endswith(".wav"):
        data.append([file,0])

df = pd.DataFrame(data, columns=["uid","class"])

df.to_csv("Data/COUGHVID_labels_audio_pitch_shift_4.csv",index=False)

print("Labels file created!")