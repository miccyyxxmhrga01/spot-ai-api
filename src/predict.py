from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from config import (
    CLASS_NAMES_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
    SPOT_CATEGORY_MAP,
    TOP_K,
)
from utils import load_class_names, pretty_label

_MODEL = None
_CLASS_NAMES = None


def get_model_and_classes():
    """Lazy-load once so the Flask API does not reload the model per request."""
    global _MODEL, _CLASS_NAMES

    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run: python src/train.py")
        _MODEL = keras.models.load_model(MODEL_PATH)

    if _CLASS_NAMES is None:
        _CLASS_NAMES = load_class_names(CLASS_NAMES_PATH)

    return _MODEL, _CLASS_NAMES


def prepare_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"Invalid or unsupported image: {error}") from error

    return np.expand_dims(array, axis=0)


def predict_image(path: str | Path, top_k: int = TOP_K) -> dict:
    model, class_names = get_model_and_classes()
    probabilities = model.predict(prepare_image(path), verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError("Model output does not match class_names.json.")

    top_k = max(1, min(top_k, len(class_names)))
    indices = np.argsort(probabilities)[::-1][:top_k]

    suggestions = []
    for index in indices:
        raw_class = class_names[int(index)]
        confidence = float(probabilities[int(index)])
        suggestions.append(
            {
                "tag": pretty_label(raw_class),
                "raw_class": raw_class,
                "spot_category": SPOT_CATEGORY_MAP.get(raw_class, pretty_label(raw_class)),
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 2),
            }
        )

    best = suggestions[0]
    recognized_talent = (
        best["raw_class"] != "other"
        and best["confidence"] >= CONFIDENCE_THRESHOLD
    )

    if best["raw_class"] == "other":
        message = "No supported visual talent was confidently identified. Ask the user to choose a category."
    elif best["confidence"] < CONFIDENCE_THRESHOLD:
        message = "Prediction is uncertain. Show suggestions, but ask the user to confirm or choose a category."
    else:
        message = "High-confidence suggestion. The user should still be allowed to edit it."

    return {
        "prediction": best["tag"],
        "raw_class": best["raw_class"],
        "spot_category": best["spot_category"],
        "confidence": best["confidence"],
        "confidence_percent": best["confidence_percent"],
        "recognized_talent": recognized_talent,
        "needs_user_confirmation": not recognized_talent,
        "threshold": CONFIDENCE_THRESHOLD,
        "message": message,
        "suggestions": suggestions,
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    result = predict_image(sys.argv[1])
    print(f"Prediction: {result['prediction']}")
    print(f"SPOT Category: {result['spot_category']}")
    print(f"Confidence: {result['confidence_percent']:.2f}%")
    print(f"Needs confirmation: {result['needs_user_confirmation']}")
    print("\nTop suggestions:")
    for suggestion in result["suggestions"]:
        print(
            f"  - {suggestion['tag']:<12} "
            f"{suggestion['confidence_percent']:>6.2f}% "
            f"({suggestion['spot_category']})"
        )


if __name__ == "__main__":
    main()
