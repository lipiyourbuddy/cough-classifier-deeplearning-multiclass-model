import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)



MODEL_PATH = "final_covidcoughnet_v3.keras"
DATA_DIR = "Data/spectrograms_split"

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# ======================
# LOAD MODEL
# ======================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded")

# ======================
# LOAD DATASET
# ======================

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)

# ======================
# TRUE LABELS
# ======================

y_true = np.concatenate([y for x,y in test_ds], axis=0)
y_true = y_true.flatten()

# ======================
# PREDICTIONS
# ======================

y_pred_probs = model.predict(test_ds)

y_pred_probs = y_pred_probs.flatten()

y_pred = (y_pred_probs > 0.5).astype(int)

# ======================
# CONFUSION MATRIX
# ======================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

# ======================
# CLASSIFICATION REPORT
# ======================

print("\nClassification Report\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

# ======================
# ROC CURVE
# ======================

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

plt.show()

print("AUC Score:", roc_auc)