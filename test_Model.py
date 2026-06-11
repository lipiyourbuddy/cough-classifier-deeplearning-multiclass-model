import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# =========================
# PATHS
# =========================

MODEL_PATH = "final_covid_cough_model.h5"
TEST_DIR = "Data/test_spectrograms"

IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("Testing spectrograms from:", TEST_DIR)
print("-" * 50)

# =========================
# TEST LOOP
# =========================

for file in os.listdir(TEST_DIR):

    if not file.endswith(".png"):
        continue

    img_path = os.path.join(TEST_DIR, file)

    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # NOTE: DO NOT normalize (/255) because training didn't

    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]

    healthy_prob = float(prediction)
    covid_prob = 1 - healthy_prob

    if covid_prob > healthy_prob:
        result = "COVID COUGH"
    else:
        result = "HEALTHY COUGH"

    print(
        f"{file} → {result} | covid: {covid_prob:.3f} | healthy: {healthy_prob:.3f}"
    )

print("\nTesting complete.")