from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import opennsfw2 as n2
from PIL import Image, UnidentifiedImageError

from config import NSFW_BLOCK_THRESHOLD, NSFW_WEIGHTS_PATH

_NSFW_MODEL = None


def get_nsfw_model():
    """Lazy-load once. The pre-trained weights download on first use if missing."""
    global _NSFW_MODEL

    if _NSFW_MODEL is None:
        _NSFW_MODEL = n2.make_open_nsfw_model(weights_path=str(NSFW_WEIGHTS_PATH))

    return _NSFW_MODEL


def nsfw_probability(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as image:
            array = n2.preprocess_image(image, n2.Preprocessing.YAHOO)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"Invalid or unsupported image: {error}") from error

    model = get_nsfw_model()
    return float(model(np.expand_dims(array, axis=0))[0][1])


def check_image(path: str | Path) -> dict:
    probability = nsfw_probability(path)
    return {
        "blocked": probability >= NSFW_BLOCK_THRESHOLD,
        "nsfw_probability": probability,
        "nsfw_threshold": NSFW_BLOCK_THRESHOLD,
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/moderation.py path/to/image.jpg")
        sys.exit(1)

    result = check_image(sys.argv[1])
    print(f"NSFW probability: {result['nsfw_probability'] * 100:.2f}%")
    print(f"Threshold: {result['nsfw_threshold'] * 100:.0f}%")
    print(f"Blocked: {result['blocked']}")


if __name__ == "__main__":
    main()
