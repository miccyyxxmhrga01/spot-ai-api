import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_class_names(class_names: list[str], path: Path) -> None:
    save_json(class_names, path)


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError("Class names not found. Train the model first: python src/train.py")
    data = load_json(path)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("class_names.json is invalid.")
    return data


def pretty_label(label: str) -> str:
    return label.replace("_", " ").title()
