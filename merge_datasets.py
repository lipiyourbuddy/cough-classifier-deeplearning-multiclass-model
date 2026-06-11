import os
import shutil

# SOURCE DATASETS
coswara_path = r"C:\Users\Admin\Python workspace\Major Project\midsem - 2nd draft\CovidCoughNet\Data\Coswara_audio"
coughvid_path = r"C:\Users\Admin\Python workspace\Major Project\midsem - 2nd draft\CovidCoughNet\Data\COUGHVID_audio"

# FINAL MERGED DATASET
final_path = r"C:\Users\Admin\Python workspace\Major Project\midsem - 2nd draft\CovidCoughNet\Data\FINAL_AUDIO"

covid_dest = os.path.join(final_path, "covid")
healthy_dest = os.path.join(final_path, "healthy")

os.makedirs(covid_dest, exist_ok=True)
os.makedirs(healthy_dest, exist_ok=True)

covid_count = 0
healthy_count = 0


def copy_files(src_folder, label, prefix):

    global covid_count, healthy_count

    files = os.listdir(src_folder)

    for f in files:

        if not f.endswith(".wav"):
            continue

        src = os.path.join(src_folder, f)

        new_name = prefix + "_" + f

        if label == "covid":
            dest = os.path.join(covid_dest, new_name)

            if not os.path.exists(dest):
                shutil.copy(src, dest)
                covid_count += 1

        elif label == "healthy":
            dest = os.path.join(healthy_dest, new_name)

            if not os.path.exists(dest):
                shutil.copy(src, dest)
                healthy_count += 1


# COPY COSWARA
copy_files(os.path.join(coswara_path, "covid"), "covid", "coswara")
copy_files(os.path.join(coswara_path, "healthy"), "healthy", "coswara")

# COPY COUGHVID
copy_files(os.path.join(coughvid_path, "covid"), "covid", "coughvid")
copy_files(os.path.join(coughvid_path, "healthy"), "healthy", "coughvid")

print("Merge complete")
print("Covid files:", covid_count)
print("Healthy files:", healthy_count)