"""Evaluate the trained SPOT model on the untouched test split.

Outputs:
- console metrics
- reports/evaluation.json
- reports/confusion_matrix.png
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow import keras

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    EVALUATION_JSON_PATH,
    IMAGE_SIZE,
    MODEL_PATH,
    REPORTS_DIR,
    TEST_DIR,
)
from dataset_audit import audit_dataset
from utils import save_json


def ensure_test_ready() -> None:
    _, report = audit_dataset()
    empty = [
        class_name
        for class_name in CLASS_NAMES
        if report["counts"].get("test", {}).get(class_name, 0) == 0
    ]
    if empty:
        raise ValueError("Test images are missing for: " + ", ".join(empty))

    if report["cross_split_duplicates"]:
        raise ValueError("Cross-split duplicates were detected. Fix them before final evaluation.")


def make_test_dataset():
    return keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        shuffle=False,
    )


def save_confusion_matrix(matrix: np.ndarray) -> None:
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True class",
        xlabel="Predicted class",
        title="SPOT Talent Classifier - Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py")

    ensure_test_ready()
    test_ds = make_test_dataset()
    model = keras.models.load_model(MODEL_PATH)

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    y_true_parts = []
    for _, labels in test_ds:
        y_true_parts.append(np.argmax(labels.numpy(), axis=1))
    y_true = np.concatenate(y_true_parts)

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    result = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": report,
        "confusion_matrix": matrix.tolist(),
        "class_order": list(CLASS_NAMES),
    }

    save_json(result, EVALUATION_JSON_PATH)
    save_confusion_matrix(matrix)

    print("\nSPOT AI - Test Evaluation")
    print("=" * 44)
    print(f"Accuracy:        {accuracy * 100:.2f}%")
    print(f"Macro precision: {macro_precision * 100:.2f}%")
    print(f"Macro recall:    {macro_recall * 100:.2f}%")
    print(f"Macro F1:        {macro_f1 * 100:.2f}%")
    print("\nPer class:")

    for class_name in CLASS_NAMES:
        item = report[class_name]
        print(
            f"  {class_name:<12} "
            f"P={item['precision'] * 100:6.2f}%  "
            f"R={item['recall'] * 100:6.2f}%  "
            f"F1={item['f1-score'] * 100:6.2f}%  "
            f"N={int(item['support'])}"
        )

    print(f"\nJSON report: {EVALUATION_JSON_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nEvaluation stopped: {error}")
        sys.exit(1)
