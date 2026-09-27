from __future__ import annotations

import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAST_N_LAYERS,
    FINE_TUNE_LEARNING_RATE,
    IMAGE_SIZE,
    INITIAL_EPOCHS,
    INITIAL_LEARNING_RATE,
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    TRAINING_HISTORY_PATH,
    TRAIN_DIR,
    VALIDATION_DIR,
)
from dataset_audit import audit_dataset, print_report
from utils import save_class_names, save_json


def ensure_training_ready() -> None:
    ok, report = audit_dataset()
    print_report(report)

    required_splits = ("train", "validation")
    empty_required = [
        f"{split}/{class_name}"
        for split in required_splits
        for class_name in CLASS_NAMES
        if report["counts"].get(split, {}).get(class_name, 0) == 0
    ]

    if empty_required:
        raise ValueError(
            "Training and validation must contain images for every class. Empty: "
            + ", ".join(empty_required)
        )

    if not ok:
        raise ValueError("Fix the dataset audit errors before training.")


def create_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        shuffle=True,
        seed=42,
    )

    validation_ds = keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), validation_ds.prefetch(autotune)


def build_model(num_classes: int):
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="talent_probabilities")(x)

    model = keras.Model(inputs, outputs, name="spot_talent_classifier")
    return model, base_model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(checkpoint_path):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def merge_histories(stage1: dict, stage2: dict) -> dict:
    merged = {}
    for key in set(stage1) | set(stage2):
        merged[key] = list(stage1.get(key, [])) + list(stage2.get(key, []))
    return merged


def main() -> None:
    print("SPOT AI - Talent Classifier Training\n")
    ensure_training_ready()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_class_names(list(CLASS_NAMES), CLASS_NAMES_PATH)

    train_ds, validation_ds = create_datasets()
    print("\nClasses:", ", ".join(CLASS_NAMES))

    model, base_model = build_model(len(CLASS_NAMES))
    compile_model(model, INITIAL_LEARNING_RATE)

    print("\nStage 1: training classification head...")
    stage1_path = MODELS_DIR / "spot_stage1_best.keras"
    stage2_path = MODELS_DIR / "spot_stage2_best.keras"

    stage1 = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=make_callbacks(stage1_path),
    )

    print("\nStage 2: fine-tuning upper MobileNetV2 layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAST_N_LAYERS]:
        layer.trainable = False

    # BatchNormalization layers are kept frozen during fine-tuning for stability
    # on smaller transfer-learning datasets.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, FINE_TUNE_LEARNING_RATE)

    stage2 = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=make_callbacks(stage2_path),
    )

    # Compare the best checkpoint from each stage. Fine-tuning is not guaranteed
    # to improve a small dataset, so keep whichever model has lower val_loss.
    stage1_model = keras.models.load_model(stage1_path)
    stage2_model = keras.models.load_model(stage2_path)
    stage1_loss, stage1_accuracy = stage1_model.evaluate(validation_ds, verbose=0)
    stage2_loss, stage2_accuracy = stage2_model.evaluate(validation_ds, verbose=0)

    if stage2_loss < stage1_loss:
        best_model = stage2_model
        selected_stage = "fine-tuned stage 2"
        val_loss, val_accuracy = stage2_loss, stage2_accuracy
    else:
        best_model = stage1_model
        selected_stage = "frozen-base stage 1"
        val_loss, val_accuracy = stage1_loss, stage1_accuracy

    best_model.save(MODEL_PATH)
    stage1_path.unlink(missing_ok=True)
    stage2_path.unlink(missing_ok=True)

    history = merge_histories(stage1.history, stage2.history)
    save_json(history, TRAINING_HISTORY_PATH)

    print(f"\nSelected model: {selected_stage}")
    print(f"Best validation loss: {val_loss:.4f}")
    print(f"Best validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Model: {MODEL_PATH}")
    print(f"History: {TRAINING_HISTORY_PATH}")
    print("\nNext step: python src/evaluate.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nTraining stopped: {error}")
        sys.exit(1)
