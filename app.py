from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from vehicle_segmentation import (
    DETECTION_COLUMNS,
    SSD_MODEL_DISPLAY_NAME,
    SSD_MODEL_NAME,
    VEHICLE_CLASS_IDS,
    choose_device,
    class_label,
    ensure_dir,
    import_yolo,
    load_torchvision_ssd,
    run_ssd_prediction,
    run_yolo_prediction,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "app"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
COMPARISON_MODELS = [
    {
        "id": "yolo26n",
        "name": "YOLO26n-seg",
        "kind": "yolo",
        "checkpoint": "yolo26n-seg.pt",
    },
    {
        "id": "yolo11n",
        "name": "YOLO11n-seg",
        "kind": "yolo",
        "checkpoint": "yolo11n-seg.pt",
    },
    {
        "id": "ssd",
        "name": SSD_MODEL_DISPLAY_NAME,
        "kind": "ssd",
        "checkpoint": SSD_MODEL_NAME,
    },
]
SUMMARY_COLUMNS = [
    "model",
    "output_type",
    "detections",
    "elapsed_seconds",
    "fps",
    "avg_confidence",
    "max_confidence",
    "classes_detected",
    "params_m",
    "device",
    "conf_threshold",
    "nms_iou",
    "note",
]
CLASS_COLORS = {
    "bicycle": (0, 114, 178),
    "car": (213, 94, 0),
    "motorcycle": (0, 158, 115),
    "bus": (204, 121, 167),
    "truck": (230, 159, 0),
}


def safe_stem(name: str) -> str:
    stem = Path(name).stem.strip().lower()
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem)
    return stem.strip("_") or "image"


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str) -> Any:
    YOLO = import_yolo()
    resolved = ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path)
    return YOLO(str(resolved))


@st.cache_resource(show_spinner=False)
def load_ssd_model() -> Any:
    return load_torchvision_ssd()


def load_comparison_models() -> tuple[dict[str, Any], dict[str, str]]:
    loaded: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for spec in COMPARISON_MODELS:
        try:
            if spec["kind"] == "yolo":
                loaded[spec["id"]] = load_yolo_model(str(spec["checkpoint"]))
            else:
                loaded[spec["id"]] = load_ssd_model()
        except Exception as exc:
            errors[spec["id"]] = str(exc)
    return loaded, errors


def load_upload(uploaded_file: Any) -> Image.Image:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def annotated_yolo_image(result: Any) -> Image.Image:
    plotted = result.plot(conf=True, labels=True, boxes=True, masks=True)
    if plotted.ndim == 3 and plotted.shape[2] == 3:
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(plotted)


def annotated_box_image(image: Image.Image, rows: list[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    line_width = max(2, min(output.size) // 240)

    for row in rows:
        label = str(row["lop"])
        color = CLASS_COLORS.get(label, (86, 180, 233))
        x1 = max(0, min(output.width - 1, int(row["x1"])))
        y1 = max(0, min(output.height - 1, int(row["y1"])))
        x2 = max(0, min(output.width - 1, int(row["x2"])))
        y2 = max(0, min(output.height - 1, int(row["y2"])))
        text = f"{label} {float(row['confidence']):.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text_box = draw.textbbox((x1, y1), text)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            [x1, text_y, x1 + text_width + 6, text_y + text_height + 4],
            fill=color,
        )
        draw.text((x1 + 3, text_y + 2), text, fill=(255, 255, 255))

    return output


def detection_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def error_summary(spec: dict[str, str], device: str, message: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": spec["name"],
        "output_type": "Mask + Box" if spec["kind"] == "yolo" else "Box",
        "detections": None,
        "elapsed_seconds": None,
        "fps": None,
        "avg_confidence": None,
        "max_confidence": None,
        "classes_detected": None,
        "params_m": None,
        "device": device,
        "note": f"Lỗi: {message}",
    }
    for class_id in VEHICLE_CLASS_IDS:
        row[f"count_{class_label(class_id).replace(' ', '_')}"] = None
    return row


def ordered_summary(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    count_columns = [
        f"count_{class_label(class_id).replace(' ', '_')}" for class_id in VEHICLE_CLASS_IDS
    ]
    columns = [column for column in [*SUMMARY_COLUMNS, *count_columns] if column in df.columns]
    return df[columns]


def save_comparison_outputs(
    uploaded_name: str,
    original: Image.Image,
    model_outputs: list[dict[str, Any]],
    comparison_table: pd.DataFrame,
    metadata: dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(OUTPUT_ROOT / f"{timestamp}_{safe_stem(uploaded_name)}")
    original.save(run_dir / "original.jpg", quality=95)
    comparison_table.to_csv(run_dir / "comparison_summary.csv", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for output in model_outputs:
        model_dir = ensure_dir(run_dir / safe_stem(str(output["model_id"])))
        output["output_image"].save(model_dir / "result.jpg", quality=95)
        detection_table(output["detections"]).to_csv(model_dir / "detections.csv", index=False)
        (model_dir / "metadata.json").write_text(
            json.dumps(output["summary"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return run_dir


def sidebar_config() -> dict[str, Any]:
    st.sidebar.header("Cấu hình")
    device_request = st.sidebar.selectbox("Thiết bị", ["auto", "mps", "cpu", "0"], index=0)
    imgsz = st.sidebar.select_slider("Kích thước ảnh YOLO", [416, 512, 640, 768, 960], value=640)
    conf = st.sidebar.slider("Confidence YOLO", 0.05, 0.90, 0.25, 0.05)
    ssd_conf = st.sidebar.slider("Confidence SSD", 0.01, 0.90, 0.15, 0.01, key="ssd_conf_threshold")
    ssd_nms_iou = st.sidebar.slider("NMS SSD", 0.10, 0.90, 0.30, 0.05)
    ssd_bicycle_fix = st.sidebar.toggle("Sửa nhãn xe đạp SSD", value=True)
    iou = st.sidebar.slider("IoU YOLO", 0.30, 0.90, 0.70, 0.05)

    class_options = {class_label(class_id): class_id for class_id in VEHICLE_CLASS_IDS}
    selected_classes = st.sidebar.multiselect(
        "Lớp xe",
        list(class_options),
        default=list(class_options),
    )
    save_result = st.sidebar.toggle("Lưu kết quả", value=True)

    return {
        "device": choose_device(device_request),
        "imgsz": imgsz,
        "conf": conf,
        "ssd_conf": ssd_conf,
        "ssd_nms_iou": ssd_nms_iou,
        "ssd_bicycle_fix": ssd_bicycle_fix,
        "iou": iou,
        "class_ids": [class_options[name] for name in selected_classes],
        "save_result": save_result,
    }


def run_model(
    spec: dict[str, str],
    model: Any,
    image: Image.Image,
    config: dict[str, Any],
) -> dict[str, Any]:
    if spec["kind"] == "yolo":
        result = run_yolo_prediction(
            model=model,
            image=image,
            model_name=spec["name"],
            class_ids=config["class_ids"],
            device=config["device"],
            imgsz=config["imgsz"],
            conf=config["conf"],
            iou=config["iou"],
        )
        output_image = annotated_yolo_image(result["raw_result"])
    else:
        result = run_ssd_prediction(
            model=model,
            image=image,
            class_ids=config["class_ids"],
            device=config["device"],
            conf=config["ssd_conf"],
            nms_iou=config["ssd_nms_iou"],
            bicycle_fix=config["ssd_bicycle_fix"],
        )
        output_image = annotated_box_image(image, result["detections"])

    result["model_id"] = spec["id"]
    result["display_name"] = spec["name"]
    result["output_image"] = output_image
    return result


def main() -> None:
    st.set_page_config(
        page_title="Vehicle Model Comparison",
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

    st.title("So Sánh YOLO26n, YOLO11n Và SSD Cho Ảnh Xe Cộ")
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

    with st.spinner("Đang tải mô hình..."):
        loaded_models, load_errors = load_comparison_models()

    for spec in COMPARISON_MODELS:
        if spec["id"] in load_errors:
            st.error(f"Không tải được {spec['name']}: {load_errors[spec['id']]}")
    if not loaded_models:
        st.stop()

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(uploaded_file.name)

        try:
            image = load_upload(uploaded_file)
        except Exception as exc:
            st.error(f"Không đọc được ảnh: {exc}")
            continue

        summary_rows: list[dict[str, Any]] = []
        model_outputs: list[dict[str, Any]] = []

        for spec in COMPARISON_MODELS:
            if spec["id"] not in loaded_models:
                summary_rows.append(error_summary(spec, config["device"], load_errors[spec["id"]]))
                continue

            with st.spinner(f"Đang chạy {spec['name']}..."):
                try:
                    output = run_model(spec, loaded_models[spec["id"]], image, config)
                except Exception as exc:
                    summary_rows.append(error_summary(spec, config["device"], str(exc)))
                    continue

            summary_rows.append(output["summary"])
            model_outputs.append(output)

        comparison_df = ordered_summary(summary_rows)
        st.dataframe(comparison_df, width="stretch", hide_index=True)
        st.download_button(
            "Tải bảng so sánh CSV",
            data=comparison_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_stem(uploaded_file.name)}_model_comparison.csv",
            mime="text/csv",
            key=f"comparison-{safe_stem(uploaded_file.name)}",
        )

        if model_outputs:
            tabs = st.tabs([output["display_name"] for output in model_outputs])
            for tab, output in zip(tabs, model_outputs):
                with tab:
                    image_cols = st.columns(2)
                    image_cols[0].image(image, caption="Ảnh gốc", width="stretch")
                    image_cols[1].image(
                        output["output_image"],
                        caption=output["output_type"],
                        width="stretch",
                    )

                    table = detection_table(output["detections"])
                    if table.empty:
                        st.warning("Không phát hiện xe ở ngưỡng hiện tại.")
                    else:
                        st.dataframe(table, width="stretch", hide_index=True)
                        st.download_button(
                            "Tải bảng CSV",
                            data=table.to_csv(index=False).encode("utf-8"),
                            file_name=f"{safe_stem(uploaded_file.name)}_{output['model_id']}_detections.csv",
                            mime="text/csv",
                            key=f"detections-{safe_stem(uploaded_file.name)}-{output['model_id']}",
                        )

        if config["save_result"]:
            metadata = {
                "input_file": uploaded_file.name,
                "device": config["device"],
                "imgsz": config["imgsz"],
                "conf": config["conf"],
                "ssd_conf": config["ssd_conf"],
                "ssd_nms_iou": config["ssd_nms_iou"],
                "ssd_bicycle_fix": config["ssd_bicycle_fix"],
                "iou": config["iou"],
                "classes": [class_label(class_id) for class_id in config["class_ids"]],
                "models": [spec["name"] for spec in COMPARISON_MODELS],
            }
            run_dir = save_comparison_outputs(
                uploaded_name=uploaded_file.name,
                original=image,
                model_outputs=model_outputs,
                comparison_table=comparison_df,
                metadata=metadata,
            )
            st.caption(f"Đã lưu: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
