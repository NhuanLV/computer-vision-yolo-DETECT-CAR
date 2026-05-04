# Báo Cáo: So Sánh YOLO26n-seg, YOLO11n-seg Và SSD Cho Ảnh Xe Cộ

## 1. Mục Tiêu

Bài toán là xử lý ảnh xe cộ trong ảnh, gồm các lớp `bicycle`, `car`, `motorcycle`, `bus`, `truck`. Kết quả chính của YOLO là mask phân vùng từng instance; kết quả của SSD là bounding box detection để so sánh tốc độ và đầu ra trên cùng ảnh.

## 2. Lý Do Chọn Mô Hình

YOLO26n-seg được chọn vì:

- là mô hình instance segmentation, đúng với yêu cầu phân vùng đối tượng;
- bản `nano` nhẹ, phù hợp Mac M1 Pro hoặc Google Colab free;
- có checkpoint pretrained trên COCO nên có thể đánh giá ngay và fine-tune nếu cần;
- Ultralytics hỗ trợ train, validation, predict, export và logging kết quả đơn giản.

Baseline so sánh là YOLO11n-seg để đánh giá lợi ích của YOLO26n-seg so với một phiên bản YOLO segmentation ổn định trước đó.

SSD SSDLite MobileNet V3 từ `torchvision` được thêm làm mô hình object detection:

- là mô hình detection phổ biến, pretrained trên COCO;
- nhẹ, phù hợp demo local;
- giúp so sánh giữa hướng segmentation và detection box-only.

## 3. Hiểu Mô Hình

YOLO là mô hình one-stage: ảnh đầu vào được xử lý qua backbone và neck để trích xuất đặc trưng, sau đó head dự đoán trực tiếp class, bounding box, confidence và mask cho từng đối tượng.

Khác biệt giữa detection và instance segmentation:

- Object detection trả về class và bounding box.
- Instance segmentation trả về thêm mask theo từng đối tượng, giúp biết hình dạng xe chính xác hơn.

Trong bài này chỉ dùng các checkpoint có hậu tố `-seg`, ví dụ `yolo26n-seg.pt`, vì checkpoint detection thường không sinh mask.

SSD là mô hình one-stage object detection. SSD chỉ dự đoán class, bounding box và confidence; mô hình này không sinh mask nên không có `mask_area_px` hoặc mask mAP.

## 4. Dữ Liệu

Dữ liệu chính là COCO-Seg, lọc 5 lớp xe:

| COCO id | Lớp |
|---:|---|
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

COCO128-Seg được dùng để sanity check pipeline. COCO val2017 được dùng cho đánh giá chính nếu đủ tài nguyên.

## 5. Thiết Lập Thí Nghiệm

Môi trường:

- Thiết bị: MacBook Pro M1 Pro hoặc Google Colab free.
- Python: 3.10 hoặc 3.11.
- Thư viện: Ultralytics, PyTorch, Torchvision, OpenCV, pycocotools, pandas, matplotlib.

Lệnh chạy chính:

```bash
python vehicle_segmentation.py compare \
  --data coco.yaml \
  --models yolo26n-seg.pt yolo11n-seg.pt \
  --batch 8 \
  --device auto
```

Metric:

- mAP50-95 mask;
- mAP50 mask;
- precision mask;
- recall mask;
- tốc độ suy luận;
- số tham số.

Với giao diện upload ảnh, bảng so sánh 3 mô hình dùng các demo metric không cần ground truth: thời gian chạy, FPS, số xe phát hiện, confidence trung bình/cao nhất, số tham số và số lượng theo lớp.

## 6. Kết Quả

Điền bảng từ `outputs/compare/comparison_summary.csv`.

| Model | mAP50-95 mask | mAP50 mask | Precision mask | Recall mask | FPS | Params (M) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg |  |  |  |  |  |  |
| YOLO11n-seg |  |  |  |  |  |  |
| SSD SSDLite MobileNet V3 | N/A | N/A | N/A | N/A |  |  |

Kết quả theo từng lớp lấy từ `outputs/compare/comparison_per_class.csv`.

| Model | bicycle | car | motorcycle | bus | truck |
|---|---:|---:|---:|---:|---:|
| YOLO26n-seg |  |  |  |  |  |
| YOLO11n-seg |  |  |  |  |  |
| SSD SSDLite MobileNet V3 | box-only | box-only | box-only | box-only | box-only |

## 7. Phân Tích

Các điểm cần phân tích:

- Lớp nào có mAP cao nhất, lớp nào thấp nhất.
- Xe nhỏ hoặc bị che khuất ảnh hưởng thế nào đến mask.
- Xe đứng sát nhau có làm mô hình tách instance sai không.
- YOLO26n-seg có nhanh hơn hoặc chính xác hơn YOLO11n-seg không trong thí nghiệm thực tế.
- SSD phát hiện box tốt trong trường hợp nào, và mất thông tin gì vì không có mask.
- Nếu fine-tune, kết quả tăng hay giảm so với pretrained và vì sao.

## 8. Minh Họa Lỗi

Chọn ảnh từ `outputs/predict/...` và phân loại lỗi:

- xe nhỏ bị bỏ sót;
- mask thiếu phần thân xe;
- mask lẫn nền đường hoặc vật thể cạnh xe;
- nhiều xe sát nhau bị gộp hoặc tách sai;
- ánh sáng kém làm giảm confidence.

## 9. Kết Luận

YOLO26n-seg phù hợp với bài toán phân vùng xe cộ nhờ tốc độ nhanh, mô hình nhẹ và hỗ trợ mask instance. Với dữ liệu COCO vehicle subset, mô hình có thể nhận diện và phân vùng nhiều loại xe phổ biến. Hạn chế chính thường nằm ở xe nhỏ, che khuất, cảnh đông xe và biên mask chưa chính xác.

## 10. Tài Liệu Tham Khảo

- Ultralytics YOLO segmentation docs: https://docs.ultralytics.com/tasks/segment/
- COCO-Seg docs: https://docs.ultralytics.com/datasets/segment/coco/
- COCO128-Seg docs: https://docs.ultralytics.com/datasets/segment/coco128-seg/
- Torchvision detection models: https://pytorch.org/vision/stable/models.html
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- PyTorch MPS docs: https://docs.pytorch.org/docs/stable/notes/mps
