# So Sánh Phương Tiện Bằng YOLO26n-seg, YOLO11n-seg Và SSD

Project này triển khai bài: tìm hiểu mô hình học sâu cho xử lý ảnh phương tiện, cài đặt và đánh giá trên tập dữ liệu công khai. Hệ thống hiện so sánh **YOLO26n-seg**, **YOLO11n-seg** và **SSD SSDLite MobileNet V3** từ `torchvision`.

Hai mô hình YOLO là instance segmentation nên trả về mask và bounding box. SSD là object detection nên chỉ trả về bounding box, confidence và lớp xe.

Các lớp xe dùng trong COCO:

| COCO id | Lớp |
|---:|---|
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

## 1. Cài Đặt Local

Nên dùng Python 3.10 hoặc 3.11. Trên Mac Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```bash
python vehicle_segmentation.py check-env
```

Nếu PyTorch nhận Apple GPU, `Selected auto device` sẽ là `mps`. Nếu chạy Colab free và có GPU, thường sẽ là CUDA device `0`.

## 2. Chạy Nhanh Với COCO128-Seg

COCO128-Seg chỉ khoảng vài MB, dùng để kiểm tra pipeline trước khi tải COCO đầy đủ.

```bash
python vehicle_segmentation.py sanity
```

Lệnh này sẽ:

- tải `data/coco128-seg`;
- chạy predict 10 ảnh mẫu và lưu ảnh mask;
- validate `yolo26n-seg.pt` trên 5 lớp xe;
- lưu kết quả vào `outputs/`.

Kết quả quan trọng:

- `outputs/predict/sanity_predict/`
- `outputs/val/sanity_val/metrics_summary.csv`
- `outputs/val/sanity_val/per_class_metrics.csv`

## 3. Predict Ảnh Xe Cộ

Sau khi đã tải COCO128-Seg, tạo trước danh sách ảnh có nhãn xe để ảnh minh họa không bị chọn ngẫu nhiên vào ảnh không có xe:

```bash
python vehicle_segmentation.py build-subset \
  --source-root data/coco128-seg \
  --output-dir outputs/subsets/coco128_vehicle \
  --train-limit 30 \
  --val-limit 10
```

Sau đó predict 10 ảnh từ danh sách này:

```bash
python vehicle_segmentation.py predict \
  --model yolo26n-seg.pt \
  --source outputs/subsets/coco128_vehicle/train.txt \
  --max-images 10 \
  --name vehicle_examples
```

Ảnh kết quả được lưu ở:

```text
outputs/predict/vehicle_examples/
```

## 4. Giao Diện So Sánh 3 Mô Hình

Dự án có giao diện web local tại `app.py`. Khi upload ảnh, hệ thống tự chạy 3 mô hình:

- `yolo26n-seg.pt`
- `yolo11n-seg.pt`
- `ssdlite320_mobilenet_v3_large` từ `torchvision`

Giao diện hiển thị bảng so sánh thời gian chạy, FPS, số xe phát hiện, confidence trung bình/cao nhất, số tham số và số lượng từng lớp xe. YOLO hiển thị mask + box; SSD hiển thị box.

```bash
streamlit run app.py
```

Nếu chưa kích hoạt môi trường:

```bash
source .venv/bin/activate
streamlit run app.py
```

Kết quả từ giao diện được lưu vào:

```text
outputs/app/
```

## 5. So Sánh Định Lượng YOLO26n-seg Và YOLO11n-seg

Chạy nhanh trên COCO128-Seg:

```bash
python vehicle_segmentation.py compare \
  --data data/coco128-seg.yaml \
  --models yolo26n-seg.pt yolo11n-seg.pt \
  --batch 4
```

Kết quả:

- `outputs/compare/comparison_summary.csv`
- `outputs/compare/comparison_per_class.csv`
- `outputs/compare/comparison_bar.png`

Lệnh CLI này dùng validation segmentation của Ultralytics nên chỉ áp dụng trực tiếp cho các mô hình YOLO `*-seg`. SSD trong `torchvision` được dùng trong giao diện upload ảnh để so sánh detection box-only trên cùng ảnh đầu vào.

## 6. Đánh Giá Chính Trên COCO Val2017

Khuyến nghị dùng lệnh chuẩn bị riêng trong project để chỉ tải COCO `val2017` và nhãn segmentation, không tải toàn bộ train/test.

```bash
python vehicle_segmentation.py prepare-coco-val
```

Sau đó đánh giá 2 mô hình trên 5 lớp xe:

```bash
python vehicle_segmentation.py compare \
  --data data/coco-val2017-seg.yaml \
  --models yolo26n-seg.pt yolo11n-seg.pt \
  --batch 8 \
  --device auto
```

Nếu dùng trực tiếp `--data coco.yaml` của Ultralytics, lệnh có thể tải COCO đầy đủ hơn 20 GB.

Các metric chính cần đưa vào báo cáo:

- `map50_95_mask`
- `map50_mask`
- `precision_mask`
- `recall_mask`
- `fps_from_inference`
- `params_m`

## 7. Fine-tune Tùy Chọn

Sau khi có COCO đầy đủ hoặc một dataset YOLO segmentation tương thích, có thể tạo danh sách ảnh chỉ chứa xe:

```bash
python vehicle_segmentation.py build-subset \
  --source-root /path/to/coco \
  --output-dir outputs/subsets/coco_vehicle \
  --train-limit 2000 \
  --val-limit 500
```

Fine-tune:

```bash
python vehicle_segmentation.py train \
  --model yolo26n-seg.pt \
  --data outputs/subsets/coco_vehicle/data.yaml \
  --epochs 25 \
  --batch 4 \
  --device auto
```

Sau đó đánh giá checkpoint tốt nhất:

```bash
python vehicle_segmentation.py validate \
  --model outputs/train/yolo26n_seg_vehicle_finetune/weights/best.pt \
  --data outputs/subsets/coco_vehicle/data.yaml \
  --name finetuned_vehicle_val
```

## 8. Notebook

Notebook có sẵn tại:

```text
notebooks/vehicle_segmentation_yolo26.ipynb
```

Notebook này chạy cùng các lệnh CLI ở trên, phù hợp để dùng local Jupyter hoặc Google Colab.

## 9. Nội Dung Báo Cáo

Mẫu báo cáo nằm ở:

```text
docs/report_template.md
```

Kết quả COCO val2017 đã chạy xong nằm ở:

```text
docs/results_coco_val2017.md
```

Trong báo cáo cần làm rõ:

- vì sao chọn YOLO26n-seg;
- vì sao thêm SSD SSDLite MobileNet V3 làm mô hình object detection so sánh;
- YOLO segmentation khác detection thường ở đâu;
- SSD không sinh mask nên không có metric mask hoặc diện tích mask;
- kết quả định lượng theo mask mAP và tốc độ;
- ảnh minh họa lỗi: xe nhỏ, che khuất, nhiều xe gần nhau, mask thiếu hoặc lẫn nền.

## 10. Nguồn Tham Khảo

- Ultralytics YOLO segmentation docs: https://docs.ultralytics.com/tasks/segment/
- COCO-Seg docs: https://docs.ultralytics.com/datasets/segment/coco/
- COCO128-Seg docs: https://docs.ultralytics.com/datasets/segment/coco128-seg/
- Torchvision detection models: https://pytorch.org/vision/stable/models.html
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- PyTorch MPS docs: https://docs.pytorch.org/docs/stable/notes/mps
