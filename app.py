from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

from vehicle_segmentation import (
    VEHICLE_CLASS_IDS,
    choose_device,
    class_label,
    ensure_dir,
    model_parameter_count,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "app"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
DEFAULT_MODELS = ["yolo26n-seg.pt", "yolo11n-seg.pt"]


def safe_stem(name: str) -> str:
    stem = Path(name).stem.strip().lower()
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem)
    return stem.strip("_") or "image"


def available_models() -> list[str]:
    models = [model for model in DEFAULT_MODELS if (ROOT / model).exists()]
    custom_outputs = sorted((ROOT / "outputs").glob("train/**/weights/best.pt"))
    models.extend(str(path.relative_to(ROOT)) for path in custom_outputs)
    return models or DEFAULT_MODELS


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(str(ROOT / model_path if not Path(model_path).is_absolute() else model_path))


def load_upload(uploaded_file: Any) -> Image.Image:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def run_prediction(
    model: YOLO,
    image: Image.Image,
    class_ids: list[int],
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> tuple[Any, float]:
    start = time.perf_counter()
    results = model.predict(
        source=np.array(image),
        classes=class_ids,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        retina_masks=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    return results[0], elapsed


def annotated_image(result: Any) -> Image.Image:
    plotted = result.plot(conf=True, labels=True, boxes=True, masks=True)
    if plotted.ndim == 3 and plotted.shape[2] == 3:
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(plotted)


def detection_table(result: Any) -> pd.DataFrame:
    columns = [
        "stt",
        "lop",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "box_area_px",
        "mask_area_px",
    ]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return pd.DataFrame(columns=columns)

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()

    mask_areas: list[int | None] = [None] * len(classes)
    masks = getattr(result, "masks", None)
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_data = masks.data.detach().cpu().numpy()
        for index in range(min(len(mask_areas), len(mask_data))):
            mask_areas[index] = int(mask_data[index].sum())

    rows: list[dict[str, Any]] = []
    for index, class_id in enumerate(classes):
        x1, y1, x2, y2 = xyxy[index]
        rows.append(
            {
                "stt": index + 1,
                "lop": class_label(int(class_id)),
                "confidence": round(float(confidences[index]), 3),
                "x1": int(round(float(x1))),
                "y1": int(round(float(y1))),
                "x2": int(round(float(x2))),
                "y2": int(round(float(y2))),
                "box_area_px": int(max(0, x2 - x1) * max(0, y2 - y1)),
                "mask_area_px": mask_areas[index],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def count_by_class(table: pd.DataFrame) -> dict[str, int]:
    if table.empty:
        return {class_label(class_id): 0 for class_id in VEHICLE_CLASS_IDS}
    counts = table["lop"].value_counts().to_dict()
    return {class_label(class_id): int(counts.get(class_label(class_id), 0)) for class_id in VEHICLE_CLASS_IDS}


def save_outputs(
    uploaded_name: str,
    original: Image.Image,
    annotated: Image.Image,
    table: pd.DataFrame,
    metadata: dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(OUTPUT_ROOT / f"{timestamp}_{safe_stem(uploaded_name)}")
    original.save(run_dir / "original.jpg", quality=95)
    annotated.save(run_dir / "result.jpg", quality=95)
    table.to_csv(run_dir / "detections.csv", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def sidebar_config() -> dict[str, Any]:
    st.sidebar.header("Cấu hình")
    model_name = st.sidebar.selectbox("Mô hình", available_models(), index=0)
    model_path = st.sidebar.text_input("Checkpoint", value=model_name)
    device_request = st.sidebar.selectbox("Thiết bị", ["auto", "mps", "cpu", "0"], index=0)
    imgsz = st.sidebar.select_slider("Kích thước ảnh", [416, 512, 640, 768, 960], value=640)
    conf = st.sidebar.slider("Confidence", 0.05, 0.90, 0.25, 0.05)
    iou = st.sidebar.slider("IoU", 0.30, 0.90, 0.70, 0.05)

    class_options = {class_label(class_id): class_id for class_id in VEHICLE_CLASS_IDS}
    selected_classes = st.sidebar.multiselect(
        "Lớp xe",
        list(class_options),
        default=list(class_options),
    )
    save_result = st.sidebar.toggle("Lưu kết quả", value=True)

    return {
        "model_path": model_path,
        "device": choose_device(device_request),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "class_ids": [class_options[name] for name in selected_classes],
        "save_result": save_result,
    }


def show_summary(table: pd.DataFrame, elapsed: float, device: str, params_m: float | None) -> None:
    counts = count_by_class(table)
    total = int(sum(counts.values()))
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    metric_cols = st.columns(4)
    metric_cols[0].metric("Tổng xe", total)
    metric_cols[1].metric("Thời gian", f"{elapsed:.2f}s")
    metric_cols[2].metric("FPS", f"{fps:.1f}")
    metric_cols[3].metric("Thiết bị", device)

    detail_cols = st.columns(5)
    for column, (name, count) in zip(detail_cols, counts.items()):
        column.metric(name, count)

    if params_m is not None:
        st.caption(f"Số tham số mô hình: {params_m:.2f}M")


def main() -> None:
    st.set_page_config(
        page_title="YOLO Vehicle Segmentation",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(49, 51, 63, 0.15); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Kiểm Tra Xe Cộ Bằng YOLO Segmentation")
    config = sidebar_config()

    if not config["class_ids"]:
        st.warning("Cần chọn ít nhất một lớp xe.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Ảnh đầu vào",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Chưa có ảnh đầu vào.")
        st.stop()

    try:
        model = load_model(config["model_path"])
    except Exception as exc:
        st.error(f"Không tải được mô hình: {exc}")
        st.stop()

    params_m = model_parameter_count(model)

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(uploaded_file.name)

        try:
            image = load_upload(uploaded_file)
        except Exception as exc:
            st.error(f"Không đọc được ảnh: {exc}")
            continue

        with st.spinner("Đang phân vùng xe..."):
            try:
                result, elapsed = run_prediction(
                    model=model,
                    image=image,
                    class_ids=config["class_ids"],
                    device=config["device"],
                    imgsz=config["imgsz"],
                    conf=config["conf"],
                    iou=config["iou"],
                )
                output_image = annotated_image(result)
                table = detection_table(result)
            except Exception as exc:
                st.error(f"Dự đoán thất bại: {exc}")
                continue

        show_summary(table, elapsed, config["device"], params_m)

        image_cols = st.columns(2)
        image_cols[0].image(image, caption="Ảnh gốc", use_container_width=True)
        image_cols[1].image(output_image, caption="Mask dự đoán", use_container_width=True)

        if table.empty:
            st.warning("Không phát hiện xe ở ngưỡng hiện tại.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button(
                "Tải bảng CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_stem(uploaded_file.name)}_detections.csv",
                mime="text/csv",
            )

        if config["save_result"]:
            metadata = {
                "input_file": uploaded_file.name,
                "model": config["model_path"],
                "device": config["device"],
                "imgsz": config["imgsz"],
                "conf": config["conf"],
                "iou": config["iou"],
                "classes": [class_label(class_id) for class_id in config["class_ids"]],
                "elapsed_seconds": elapsed,
                "detections": len(table),
            }
            run_dir = save_outputs(uploaded_file.name, image, output_image, table, metadata)
            st.caption(f"Đã lưu: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
