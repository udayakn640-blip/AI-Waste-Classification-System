"""
Train an image classifier for five waste categories.

Expected dataset structure:
dataset/
  organic/
  paper/
  plastic/
  glass/
  metal/

Place labelled JPG/PNG images inside each category folder, then run:
    python train_model.py
"""
from pathlib import Path
import json
import tensorflow as tf

DATA_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "waste_classifier.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 25

def main():
    if not DATA_DIR.exists():
        raise SystemExit(
            "Dataset folder not found. Create dataset/ with subfolders: "
            "organic, paper, plastic, glass, metal. See README.md."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("Please provide at least two category folders with images.")
    print("Detected categories:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save the class ordering used by the model.
    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    final_train_acc = history.history.get("accuracy", [None])[-1]
    final_val_acc = history.history.get("val_accuracy", [None])[-1]
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved class names to: {LABELS_PATH}")
    print(f"Last epoch training accuracy: {final_train_acc}")
    print(f"Last epoch validation accuracy: {final_val_acc}")
    print("These are training/validation metrics, not independent test-set results.")

if __name__ == "__main__":
    main()
