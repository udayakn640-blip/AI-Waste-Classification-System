from pathlib import Path
import json
import tensorflow as tf

DATA_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "waste_classifier.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 123
EPOCHS = 15


def main():
    if not DATA_DIR.exists():
        raise SystemExit("Dataset folder not found.")

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
        raise SystemExit("At least two category folders are required.")

    print("Detected categories:", class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(
        1000, seed=SEED
    ).prefetch(buffer_size=autotune)

    val_ds = val_ds.cache().prefetch(
        buffer_size=autotune
    )

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    # Freeze pretrained layers during initial training.
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=IMAGE_SIZE + (3,)
    )

    x = augmentation(inputs)

    # MobileNetV2 expects pixel values scaled to [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        len(class_names),
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save the exact class order used by the model.
    LABELS_PATH.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    print("\nModel saved to:", MODEL_PATH)
    print("Class names saved to:", LABELS_PATH)
    print(
        "Best validation accuracy:",
        max(history.history["val_accuracy"]),
    )
    print(
        "Best training accuracy:",
        max(history.history["accuracy"]),
    )


if __name__ == "__main__":
    main()