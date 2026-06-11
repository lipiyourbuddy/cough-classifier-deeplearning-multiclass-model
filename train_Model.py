import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


TRAIN_DIR   = "Data/spectrograms_split/train"
VAL_DIR     = "Data/spectrograms_split/val"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_P1   = 30
EPOCHS_P2   = 20
SEED        = 42

CLASS_NAMES = ["asthma", "covid", "healthy", "other_resp"]
NUM_CLASSES = len(CLASS_NAMES)

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

print("Classes:", train_ds_raw.class_names)


labels = np.concatenate([y for _, y in train_ds_raw], axis=0)
cw     = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(cw))
print("Class weights:", {CLASS_NAMES[k]: round(v, 3) for k, v in class_weights.items()})


AUTOTUNE = tf.data.AUTOTUNE

train_ds = (train_ds_raw
            .cache()
            .shuffle(2000, seed=SEED)
            .prefetch(AUTOTUNE))

val_ds = (val_ds_raw
          .cache()
          .prefetch(AUTOTUNE))


inputs     = tf.keras.Input(shape=(224, 224, 3), name="input")
base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=inputs
)
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# PHASE 1 — Train head only, base frozen

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

p1_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_v6_phase1.keras",
        monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    )
]

print("\n" + "="*50)
print("PHASE 1: Training head layers (base frozen)")
print("="*50)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P1,
    class_weight=class_weights,
    callbacks=p1_callbacks
)

# PHASE 2 — Unfreeze top 20 layers, fine-tune

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"\nFine-tuning {trainable_count} layers of base model")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

p2_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_v6_phase2.keras",
        monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3,
        patience=4, min_lr=1e-7, verbose=1
    )
]

print("\n" + "="*50)
print("PHASE 2: Fine-tuning top layers")
print("="*50)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P2,
    class_weight=class_weights,
    callbacks=p2_callbacks
)


model.save("final_covidcoughnet_v6.keras")
print("\nFinal model saved → final_covidcoughnet_v6.keras")


print("\nRunning evaluation on validation set...")
y_true, y_pred_probs = [], []

for x_batch, y_batch in val_ds:
    probs = model.predict(x_batch, verbose=0)
    y_pred_probs.extend(probs)
    y_true.extend(y_batch.numpy())

y_true       = np.array(y_true)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_pred_probs = np.array(y_pred_probs)

auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
print(f"\nMacro AUC (one-vs-rest): {auc:.4f}")

print("\n── Classification Report ──────────────────────")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5)
plt.title("Confusion Matrix — CovidCoughNet v3", fontsize=13)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix_v3.png", dpi=150)
plt.show()
print("Confusion matrix saved → confusion_matrix_v3.png")


def plot_metric(metric, title, fname):
    p1  = history1.history.get(metric, [])
    p2  = history2.history.get(metric, [])
    vp1 = history1.history.get(f"val_{metric}", [])
    vp2 = history2.history.get(f"val_{metric}", [])
    combined     = p1 + p2
    val_combined = vp1 + vp2
    split        = len(p1)
    plt.figure(figsize=(9, 4))
    plt.plot(combined,     label="Train",      linewidth=2)
    plt.plot(val_combined, label="Validation", linewidth=2)
    plt.axvline(split, color='gray', linestyle='--',
                linewidth=1.2, label="Fine-tune start")
    plt.title(title, fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

plot_metric("accuracy", "Accuracy", "curve_accuracy_v3.png")
plot_metric("loss",     "Loss",     "curve_loss_v3.png")

print("\nAll curves saved.")
print("Done. Check confusion_matrix_v3.png and classification report above.")