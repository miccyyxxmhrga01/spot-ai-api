"""Audit the SPOT image dataset before training.

Checks:
- required split/class folders
- image counts and class imbalance
- unreadable/corrupt images
- exact duplicate files, especially duplicates crossing train/validation/test

Run:
    python src/dataset_audit.py
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import (
    CLASS_NAMES,
    MIN_RECOMMENDED_IMAGES_PER_CLASS,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
)


@dataclass
class ImageRecord:
    split: str
    class_name: str
    path: Path
    sha256: str


def iter_image_files(folder: Path):
    if not folder.exists():
        return
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return str(error)


def audit_dataset() -> tuple[bool, dict]:
    splits = {
        "train": TRAIN_DIR,
        "validation": VALIDATION_DIR,
        "test": TEST_DIR,
    }

    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    bad_images: list[dict] = []
    records: list[ImageRecord] = []

    expected_classes = set(CLASS_NAMES)

    for split_name, split_dir in splits.items():
        if not split_dir.exists():
            errors.append(f"Missing split folder: {split_dir}")
            counts[split_name] = {}
            continue

        actual_classes = {
            path.name
            for path in split_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }

        missing = sorted(expected_classes - actual_classes)
        unexpected = sorted(actual_classes - expected_classes)

        if missing:
            errors.append(f"{split_name}: missing class folders: {', '.join(missing)}")
        if unexpected:
            errors.append(f"{split_name}: unexpected class folders: {', '.join(unexpected)}")

        counts[split_name] = {}

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            files = list(iter_image_files(class_dir) or [])
            counts[split_name][class_name] = len(files)

            if len(files) == 0:
                warnings.append(f"{split_name}/{class_name}: no images yet")

            for path in files:
                problem = validate_image(path)
                if problem:
                    bad_images.append({
                        "split": split_name,
                        "class": class_name,
                        "path": str(path),
                        "error": problem,
                    })
                    continue

                records.append(ImageRecord(
                    split=split_name,
                    class_name=class_name,
                    path=path,
                    sha256=sha256_file(path),
                ))

    # Training-data guidance.
    for class_name in CLASS_NAMES:
        train_count = counts.get("train", {}).get(class_name, 0)
        if 0 < train_count < MIN_RECOMMENDED_IMAGES_PER_CLASS:
            warnings.append(
                f"train/{class_name}: only {train_count} image(s); "
                f"aim for at least {MIN_RECOMMENDED_IMAGES_PER_CLASS}, preferably 300-500+ diverse images"
            )

    train_counts = [counts.get("train", {}).get(name, 0) for name in CLASS_NAMES]
    nonzero_train_counts = [count for count in train_counts if count > 0]
    if len(nonzero_train_counts) >= 2:
        smallest = min(nonzero_train_counts)
        largest = max(nonzero_train_counts)
        if largest > smallest * 1.5:
            warnings.append(
                f"Training classes are imbalanced ({smallest} to {largest} images). "
                "Try to keep class sizes reasonably close."
            )

    if bad_images:
        errors.append(f"Found {len(bad_images)} unreadable/corrupt image(s).")

    hashes: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        hashes[record.sha256].append(record)

    duplicate_groups = []
    cross_split_duplicates = []

    for digest, items in hashes.items():
        if len(items) < 2:
            continue

        group = {
            "sha256": digest,
            "files": [str(item.path) for item in items],
            "splits": sorted({item.split for item in items}),
            "classes": sorted({item.class_name for item in items}),
        }
        duplicate_groups.append(group)

        if len(group["splits"]) > 1:
            cross_split_duplicates.append(group)

    if cross_split_duplicates:
        errors.append(
            f"Found {len(cross_split_duplicates)} exact duplicate group(s) crossing dataset splits. "
            "This can leak training examples into validation/test results."
        )
    elif duplicate_groups:
        warnings.append(f"Found {len(duplicate_groups)} exact duplicate group(s) within splits.")

    total_images = sum(
        count
        for split_counts in counts.values()
        for count in split_counts.values()
    )

    report = {
        "ok": len(errors) == 0,
        "total_images": total_images,
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
        "bad_images": bad_images,
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicates": cross_split_duplicates,
    }

    return len(errors) == 0, report


def print_report(report: dict) -> None:
    print("SPOT AI - Dataset Audit")
    print("=" * 52)

    for split, split_counts in report["counts"].items():
        print(f"\n{split.upper()}")
        for class_name, count in split_counts.items():
            print(f"  {class_name:<12} {count:>5}")

    print(f"\nTotal images: {report['total_images']}")

    if report["errors"]:
        print("\nERRORS")
        for error in report["errors"]:
            print(f"  - {error}")

    if report["warnings"]:
        print("\nWARNINGS")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["cross_split_duplicates"]:
        print("\nCROSS-SPLIT DUPLICATES")
        for group in report["cross_split_duplicates"][:10]:
            print("  - " + " | ".join(group["files"]))

    if report["ok"] and report["total_images"] == 0:
        print("\nFolder structure is valid, but the dataset is still empty.")
    elif report["ok"]:
        print("\nAudit passed with no blocking errors.")
    else:
        print("\nAudit found problems that should be fixed before training.")


def main() -> None:
    ok, report = audit_dataset()
    print_report(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
