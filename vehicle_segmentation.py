#!/usr/bin/env python3
"""YOLO vehicle instance segmentation experiments.

This CLI keeps the coursework workflow reproducible on local machines and
Google Colab:

1. Check the runtime device.
2. Download a small public COCO128-Seg sanity dataset.
3. Predict vehicle masks on sample images.
4. Validate YOLO segmentation models on a vehicle-only COCO class subset.
5. Optionally fine-tune on a small vehicle subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable


COCO128_SEG_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "coco128-seg.zip"
)
COCO2017_LABELS_SEG_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "coco2017labels-segments.zip"
)
COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"

COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_class_ids(value: str | None) -> list[int]:
    if not value:
        return VEHICLE_CLASS_IDS.copy()

    aliases = {
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "truck": 7,
    }
    class_ids: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in aliases:
            class_ids.append(aliases[item])
        else:
            class_ids.append(int(item))
    return class_ids


def class_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_NAMES):
        return COCO_NAMES[class_id]
    return f"class_{class_id}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "PyTorch is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc


def import_yolo() -> Any:
    try:
        from ultralytics import YOLO

        return YOLO
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "Ultralytics is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    torch = import_torch()
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    number = scalar(value)
    if number is not None:
        return number
    return str(value)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def yaml_quote(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def write_coco_yaml(path: Path, root: Path, train: str, val: str, test: str = "") -> None:
    ensure_dir(path.parent)
    lines = [
        f"path: {yaml_quote(root.resolve())}",
        f"train: {yaml_quote(train)}",
        f"val: {yaml_quote(val)}",
        f"test: {yaml_quote(test)}",
        "names:",
    ]
    for index, name in enumerate(COCO_NAMES):
        lines.append(f"  {index}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def download_file(url: str, path: Path) -> None:
    ensure_dir(path.parent)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, path.open("wb") as file:
        total = int(response.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}% ({downloaded / 1e6:.1f} MB)", end="")
        print()


def collect_images(source: str, max_images: int | None = None, seed: int = 42) -> list[str] | str:
    source_path = Path(source)
    if not source_path.exists():
        return source

    if source_path.is_file():
        if source_path.suffix.lower() != ".txt":
            return source

        images = [
            line.strip()
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and Path(line.strip()).suffix.lower() in IMAGE_SUFFIXES
        ]
        if max_images and len(images) > max_images:
            rng = random.Random(seed)
            images = rng.sample(images, max_images)
            images.sort()
        return images

    images = sorted(
        str(path)
        for path in source_path.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_images and len(images) > max_images:
        rng = random.Random(seed)
        images = rng.sample(images, max_images)
        images.sort()
    return images


def model_parameter_count(model: Any) -> float | None:
    try:
        return sum(param.numel() for param in model.model.parameters()) / 1_000_000
    except Exception:
        return None


def metrics_summary_row(
    metrics: Any,
    model_name: str,
    data: str,
    device: str,
    elapsed_seconds: float,
    params_m: float | None,
) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    seg = getattr(metrics, "seg", None)
    speed = getattr(metrics, "speed", {}) or {}

    row: dict[str, Any] = {
        "model": model_name,
        "data": data,
        "device": device,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "params_m": round(params_m, 3) if params_m is not None else None,
        "precision_mask": scalar(getattr(seg, "mp", None)),
        "recall_mask": scalar(getattr(seg, "mr", None)),
        "map50_mask": scalar(getattr(seg, "map50", None)),
        "map75_mask": scalar(getattr(seg, "map75", None)),
        "map50_95_mask": scalar(getattr(seg, "map", None)),
        "precision_box": scalar(getattr(box, "mp", None)),
        "recall_box": scalar(getattr(box, "mr", None)),
        "map50_box": scalar(getattr(box, "map50", None)),
        "map50_95_box": scalar(getattr(box, "map", None)),
    }

    for key, value in speed.items():
        numeric = scalar(value)
        if numeric is not None:
            row[f"speed_{key}_ms"] = numeric

    inference_ms = scalar(speed.get("inference")) if isinstance(speed, dict) else None
    if inference_ms and inference_ms > 0:
        row["fps_from_inference"] = 1000.0 / inference_ms

    return row


def per_class_rows(metrics: Any, model_name: str, class_ids: list[int]) -> list[dict[str, Any]]:
    seg = getattr(metrics, "seg", None)
    box = getattr(metrics, "box", None)
    seg_map_values = getattr(seg, "maps", None)
    box_map_values = getattr(box, "maps", None)
    seg_maps = list(seg_map_values) if seg_map_values is not None else []
    box_maps = list(box_map_values) if box_map_values is not None else []

    rows: list[dict[str, Any]] = []
    for offset, class_id in enumerate(class_ids):
        if len(seg_maps) == len(COCO_NAMES):
            seg_map = scalar(seg_maps[class_id])
        elif offset < len(seg_maps):
            seg_map = scalar(seg_maps[offset])
        else:
            seg_map = None

        if len(box_maps) == len(COCO_NAMES):
            box_map = scalar(box_maps[class_id])
        elif offset < len(box_maps):
            box_map = scalar(box_maps[offset])
        else:
            box_map = None

        rows.append(
            {
                "model": model_name,
                "class_id": class_id,
                "class_name": class_label(class_id),
                "map50_95_mask": seg_map,
                "map50_95_box": box_map,
            }
        )
    return rows


def validate_model(
    model_name: str,
    data: str,
    output_dir: Path,
    device: str,
    class_ids: list[int],
    imgsz: int,
    batch: int,
    split: str,
    name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    YOLO = import_yolo()
    run_name = name or Path(model_name).stem
    run_dir = ensure_dir((output_dir / "val").resolve())

    print(f"Validating {model_name} on {data}")
    print(f"Classes: {', '.join(class_label(item) for item in class_ids)}")

    model = YOLO(model_name)
    params_m = model_parameter_count(model)
    start = time.time()
    metrics = model.val(
        data=data,
        split=split,
        classes=class_ids,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(run_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    elapsed = time.time() - start

    summary = metrics_summary_row(
        metrics=metrics,
        model_name=model_name,
        data=data,
        device=device,
        elapsed_seconds=elapsed,
        params_m=params_m,
    )
    class_rows = per_class_rows(metrics, model_name, class_ids)

    result_dir = run_dir / run_name
    write_json(result_dir / "metrics_summary.json", summary)
    write_csv(result_dir / "metrics_summary.csv", [summary])
    write_csv(result_dir / "per_class_metrics.csv", class_rows)
    if hasattr(metrics, "results_dict"):
        write_json(result_dir / "raw_results_dict.json", metrics.results_dict)

    return summary, class_rows


def make_comparison_plot(csv_path: Path, image_path: Path) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping plot because pandas/matplotlib is not installed.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    columns = [
        column
        for column in ["map50_95_mask", "map50_mask", "map50_95_box"]
        if column in df.columns
    ]
    if not columns:
        return

    ax = df.set_index("model")[columns].plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Vehicle segmentation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    ensure_dir(image_path.parent)
    plt.savefig(image_path, dpi=160)
    plt.close()


def command_check_env(_: argparse.Namespace) -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Working directory: {Path.cwd()}")

    try:
        torch = import_torch()
        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"MPS available: {mps_available}")
        print(f"Selected auto device: {choose_device('auto')}")
    except Exception as exc:
        print(f"torch: unavailable ({exc})")

    try:
        import ultralytics

        print(f"ultralytics: {ultralytics.__version__}")
    except Exception as exc:
        print(f"ultralytics: unavailable ({exc})")


def command_prepare_coco128(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    dataset_root = output_root / "coco128-seg"
    yaml_path = output_root / "coco128-seg.yaml"
    zip_path = Path(args.download_dir) / "coco128-seg.zip"

    if args.force and dataset_root.exists():
        import shutil

        shutil.rmtree(dataset_root)

    if not dataset_root.exists():
        download_file(COCO128_SEG_URL, zip_path)
        ensure_dir(output_root)
        safe_extract(zip_path, output_root)

    if not dataset_root.exists():
        raise RuntimeError(f"Expected dataset at {dataset_root}, but it was not found.")

    write_coco_yaml(
        path=yaml_path,
        root=dataset_root,
        train="images/train2017",
        val="images/train2017",
    )
    print(f"Dataset ready: {dataset_root}")
    print(f"Dataset YAML: {yaml_path}")


def command_prepare_coco_val(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    dataset_root = output_root / "coco"
    image_dir = dataset_root / "images" / "val2017"
    label_dir = dataset_root / "labels" / "val2017"
    download_dir = Path(args.download_dir)
    yaml_path = Path(args.yaml_path)

    if not label_dir.exists() or not any(label_dir.glob("*.txt")):
        labels_zip = download_dir / "coco2017labels-segments.zip"
        if not labels_zip.exists():
            download_file(COCO2017_LABELS_SEG_URL, labels_zip)
        safe_extract(labels_zip, output_root)

    if not image_dir.exists() or not any(image_dir.glob("*.jpg")):
        images_zip = download_dir / "val2017.zip"
        if not images_zip.exists():
            download_file(COCO_VAL2017_URL, images_zip)
        ensure_dir(dataset_root / "images")
        safe_extract(images_zip, dataset_root / "images")

    if not image_dir.exists() or not any(image_dir.glob("*.jpg")):
        raise RuntimeError(f"Expected COCO val images at {image_dir}, but they were not found.")
    if not label_dir.exists() or not any(label_dir.glob("*.txt")):
        raise RuntimeError(f"Expected COCO segment labels at {label_dir}, but they were not found.")

    write_coco_yaml(
        path=yaml_path,
        root=dataset_root,
        train="images/val2017",
        val="images/val2017",
    )

    image_count = len(list(image_dir.glob("*.jpg")))
    label_count = len(list(label_dir.glob("*.txt")))
    metadata = {
        "dataset_root": str(dataset_root.resolve()),
        "yaml_path": str(yaml_path.resolve()),
        "image_count": image_count,
        "label_count": label_count,
        "images": str(image_dir.resolve()),
        "labels": str(label_dir.resolve()),
    }
    write_json(yaml_path.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def command_predict(args: argparse.Namespace) -> None:
    YOLO = import_yolo()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    device = choose_device(args.device)
    class_ids = parse_class_ids(args.classes)
    source = collect_images(args.source, args.max_images, args.seed)
    source_list = source if isinstance(source, list) else None

    model = YOLO(args.model)
    start = time.time()
    results = model.predict(
        source=source,
        classes=class_ids,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=str(output_dir / "predict"),
        name=args.name,
        exist_ok=True,
        save=True,
        retina_masks=True,
        verbose=True,
    )
    elapsed = time.time() - start

    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        counts = {class_label(class_id): 0 for class_id in class_ids}
        if result.boxes is not None and result.boxes.cls is not None:
            for class_value in result.boxes.cls.tolist():
                class_id = int(class_value)
                if class_id in class_ids:
                    counts[class_label(class_id)] = counts.get(class_label(class_id), 0) + 1

        row: dict[str, Any] = {
            "image": source_list[index] if source_list and index < len(source_list) else getattr(result, "path", ""),
            "detections": sum(counts.values()),
            "has_masks": result.masks is not None,
            "elapsed_seconds_total": round(elapsed, 3),
            "device": device,
            "model": args.model,
        }
        for name, count in counts.items():
            row[f"count_{name.replace(' ', '_')}"] = count
        for key, value in (getattr(result, "speed", {}) or {}).items():
            numeric = scalar(value)
            if numeric is not None:
                row[f"speed_{key}_ms"] = numeric
        rows.append(row)

    result_dir = output_dir / "predict" / args.name
    write_csv(result_dir / "prediction_summary.csv", rows)
    print(f"Prediction outputs: {result_dir}")


def command_validate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    class_ids = parse_class_ids(args.classes)
    validate_model(
        model_name=args.model,
        data=args.data,
        output_dir=Path(args.output_dir),
        device=device,
        class_ids=class_ids,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        name=args.name,
    )


def command_compare(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    class_ids = parse_class_ids(args.classes)
    root_output_dir = Path(args.output_dir).resolve()
    output_dir = ensure_dir(root_output_dir / "compare")

    summaries: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    for model_name in args.models:
        summary, class_rows = validate_model(
            model_name=model_name,
            data=args.data,
            output_dir=root_output_dir,
            device=device,
            class_ids=class_ids,
            imgsz=args.imgsz,
            batch=args.batch,
            split=args.split,
            name=f"{Path(model_name).stem}_{args.name_suffix}",
        )
        summaries.append(summary)
        per_class.extend(class_rows)

    summary_csv = output_dir / "comparison_summary.csv"
    write_csv(summary_csv, summaries)
    write_csv(output_dir / "comparison_per_class.csv", per_class)
    make_comparison_plot(summary_csv, output_dir / "comparison_bar.png")
    print(f"Comparison outputs: {output_dir}")


def command_sanity(args: argparse.Namespace) -> None:
    output_root = Path(args.data_root)
    yaml_path = output_root / "coco128-seg.yaml"
    if not yaml_path.exists():
        prepare_args = argparse.Namespace(
            output_root=str(output_root),
            download_dir=str(Path(args.output_dir) / "downloads"),
            force=False,
        )
        command_prepare_coco128(prepare_args)

    image_dir = output_root / "coco128-seg" / "images" / "train2017"
    predict_args = argparse.Namespace(
        model=args.model,
        source=str(image_dir),
        max_images=args.max_images,
        seed=args.seed,
        classes=args.classes,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        output_dir=args.output_dir,
        name="sanity_predict",
    )
    command_predict(predict_args)

    validate_args = argparse.Namespace(
        model=args.model,
        data=str(yaml_path),
        classes=args.classes,
        imgsz=args.imgsz,
        batch=args.batch,
        split="val",
        device=args.device,
        output_dir=args.output_dir,
        name="sanity_val",
    )
    command_validate(validate_args)


def label_has_class(label_path: Path, class_ids: set[int]) -> bool:
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if int(float(parts[0])) in class_ids:
                return True
    except Exception:
        return False
    return False


def image_for_label(root: Path, split: str, label_path: Path) -> Path | None:
    image_dir = root / "images" / split
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{label_path.stem}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    matches = list(image_dir.glob(f"{label_path.stem}.*"))
    for match in matches:
        if match.suffix.lower() in IMAGE_SUFFIXES:
            return match.resolve()
    return None


def subset_for_split(
    root: Path,
    split: str,
    class_ids: set[int],
    limit: int | None,
    seed: int,
) -> list[Path]:
    label_dir = root / "labels" / split
    if not label_dir.exists():
        raise RuntimeError(f"Label directory not found: {label_dir}")

    images: list[Path] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        if not label_has_class(label_path, class_ids):
            continue
        image_path = image_for_label(root, split, label_path)
        if image_path:
            images.append(image_path)

    if limit and len(images) > limit:
        rng = random.Random(seed)
        images = rng.sample(images, limit)
        images.sort()
    return images


def write_image_list(path: Path, images: Iterable[Path]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")


def command_build_subset(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_dir = ensure_dir(Path(args.output_dir))
    class_ids = set(parse_class_ids(args.classes))

    val_split = args.val_split
    if not (source_root / "labels" / val_split).exists():
        print(f"Validation split {val_split} not found. Reusing {args.train_split}.")
        val_split = args.train_split

    train_images = subset_for_split(
        source_root,
        args.train_split,
        class_ids,
        args.train_limit,
        args.seed,
    )
    val_images = subset_for_split(
        source_root,
        val_split,
        class_ids,
        args.val_limit,
        args.seed,
    )

    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    yaml_path = output_dir / "data.yaml"
    write_image_list(train_txt, train_images)
    write_image_list(val_txt, val_images)
    write_coco_yaml(
        path=yaml_path,
        root=source_root,
        train=str(train_txt.resolve()),
        val=str(val_txt.resolve()),
    )

    metadata = {
        "source_root": str(source_root),
        "classes": sorted(class_ids),
        "class_names": [class_label(class_id) for class_id in sorted(class_ids)],
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_list": str(train_txt.resolve()),
        "val_list": str(val_txt.resolve()),
        "data_yaml": str(yaml_path.resolve()),
    }
    write_json(output_dir / "subset_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def command_train(args: argparse.Namespace) -> None:
    YOLO = import_yolo()
    device = choose_device(args.device)
    class_ids = parse_class_ids(args.classes)
    output_dir = ensure_dir(Path(args.output_dir).resolve() / "train")

    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "classes": class_ids,
        "project": str(output_dir),
        "name": args.name,
        "exist_ok": True,
        "seed": args.seed,
        "patience": args.patience,
        "plots": True,
    }
    if args.fraction is not None:
        train_kwargs["fraction"] = args.fraction
    if args.cache:
        train_kwargs["cache"] = args.cache

    print(f"Training {args.model} on {args.data}")
    print(f"Device: {device}")
    print(f"Classes: {', '.join(class_label(item) for item in class_ids)}")
    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", output_dir / args.name))
    write_json(
        save_dir / "train_command.json",
        {
            "model": args.model,
            "data": args.data,
            "kwargs": train_kwargs,
            "best_weights": str(save_dir / "weights" / "best.pt"),
        },
    )
    print(f"Training outputs: {save_dir}")


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--classes", default=",".join(str(item) for item in VEHICLE_CLASS_IDS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vehicle instance segmentation with YOLO26/YOLO11.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_env = subparsers.add_parser("check-env", help="Show Python, torch, device, and Ultralytics info.")
    check_env.set_defaults(func=command_check_env)

    prepare = subparsers.add_parser("prepare-coco128", help="Download COCO128-Seg into data/.")
    prepare.add_argument("--output-root", default="data")
    prepare.add_argument("--download-dir", default="outputs/downloads")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare_coco128)

    prepare_val = subparsers.add_parser(
        "prepare-coco-val",
        help="Download only COCO val2017 images and segmentation labels.",
    )
    prepare_val.add_argument("--output-root", default="datasets")
    prepare_val.add_argument("--download-dir", default="outputs/downloads")
    prepare_val.add_argument("--yaml-path", default="data/coco-val2017-seg.yaml")
    prepare_val.set_defaults(func=command_prepare_coco_val)

    predict = subparsers.add_parser("predict", help="Predict vehicle masks and save visualizations.")
    add_common_model_args(predict)
    predict.add_argument("--source", default="https://ultralytics.com/images/bus.jpg")
    predict.add_argument("--max-images", type=int, default=10)
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.7)
    predict.add_argument("--name", default="vehicle_predict")
    predict.set_defaults(func=command_predict)

    validate = subparsers.add_parser("validate", help="Validate one segmentation model on vehicle classes.")
    add_common_model_args(validate)
    validate.add_argument("--data", default="coco.yaml")
    validate.add_argument("--batch", type=int, default=4)
    validate.add_argument("--split", default="val")
    validate.add_argument("--name", default=None)
    validate.set_defaults(func=command_validate)

    compare = subparsers.add_parser("compare", help="Compare two or more segmentation models.")
    compare.add_argument("--models", nargs="+", default=["yolo26n-seg.pt", "yolo11n-seg.pt"])
    compare.add_argument("--data", default="coco.yaml")
    compare.add_argument("--classes", default=",".join(str(item) for item in VEHICLE_CLASS_IDS))
    compare.add_argument("--imgsz", type=int, default=640)
    compare.add_argument("--batch", type=int, default=4)
    compare.add_argument("--split", default="val")
    compare.add_argument("--device", default="auto")
    compare.add_argument("--output-dir", default="outputs")
    compare.add_argument("--name-suffix", default="vehicle")
    compare.set_defaults(func=command_compare)

    sanity = subparsers.add_parser("sanity", help="Run quick download, prediction, and validation on COCO128-Seg.")
    add_common_model_args(sanity)
    sanity.add_argument("--data-root", default="data")
    sanity.add_argument("--max-images", type=int, default=10)
    sanity.add_argument("--seed", type=int, default=42)
    sanity.add_argument("--conf", type=float, default=0.25)
    sanity.add_argument("--iou", type=float, default=0.7)
    sanity.add_argument("--batch", type=int, default=4)
    sanity.set_defaults(func=command_sanity)

    subset = subparsers.add_parser("build-subset", help="Create train/val image lists containing vehicle labels.")
    subset.add_argument("--source-root", required=True)
    subset.add_argument("--output-dir", default="outputs/subsets/coco_vehicle")
    subset.add_argument("--classes", default=",".join(str(item) for item in VEHICLE_CLASS_IDS))
    subset.add_argument("--train-split", default="train2017")
    subset.add_argument("--val-split", default="val2017")
    subset.add_argument("--train-limit", type=int, default=2000)
    subset.add_argument("--val-limit", type=int, default=500)
    subset.add_argument("--seed", type=int, default=42)
    subset.set_defaults(func=command_build_subset)

    train = subparsers.add_parser("train", help="Fine-tune a YOLO segmentation model on vehicle classes.")
    add_common_model_args(train)
    train.add_argument("--data", required=True)
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--name", default="yolo26n_seg_vehicle_finetune")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--fraction", type=float, default=None)
    train.add_argument("--cache", default=None, choices=[None, "ram", "disk"])
    train.set_defaults(func=command_train)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
