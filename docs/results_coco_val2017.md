# Kết Quả COCO Val2017: Vehicle Instance Segmentation

Đánh giá trên COCO val2017, lọc 5 lớp xe: `bicycle`, `car`, `motorcycle`, `bus`, `truck`.

## Kết Quả Tổng Hợp

| Model | Mask mAP50-95 | Mask mAP50 | Mask Precision | Mask Recall | Box mAP50-95 | Inference FPS | Thời gian eval |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg | 0.3248 | 0.5389 | 0.6730 | 0.4797 | 0.4038 | 55.26 | 466.1s |
| YOLO11n-seg | 0.3176 | 0.5393 | 0.6827 | 0.4817 | 0.3986 | 111.91 | 911.8s |

Ghi chú: `Inference FPS` được tính từ thời gian inference của Ultralytics, chưa bao gồm postprocess mask. Trên MPS, YOLO11n-seg có inference nhanh hơn nhưng postprocess mask chậm hơn đáng kể, nên tổng thời gian evaluation dài hơn YOLO26n-seg.

## Kết Quả Theo Lớp

| Model | bicycle | car | motorcycle | bus | truck |
|---|---:|---:|---:|---:|---:|
| YOLO26n-seg mask mAP50-95 | 0.1505 | 0.3000 | 0.3149 | 0.5871 | 0.2716 |
| YOLO11n-seg mask mAP50-95 | 0.1526 | 0.2928 | 0.3104 | 0.5784 | 0.2538 |

## Nhận Xét Ngắn

- YOLO26n-seg có `mask mAP50-95` cao hơn YOLO11n-seg khoảng 0.0072 điểm, phù hợp chọn làm mô hình chính.
- `bus` là lớp dễ nhất trong nhóm xe, do kích thước lớn và hình dạng rõ; cả hai mô hình đều đạt mask mAP50-95 khoảng 0.58.
- `bicycle` là lớp khó nhất, mask mAP50-95 chỉ khoảng 0.15 do đối tượng mảnh, nhiều chi tiết nhỏ và dễ bị che khuất.
- `car` có nhiều instance nhất nhưng mask mAP50-95 trung bình, vì nhiều xe nhỏ, bị che khuất hoặc nằm trong cảnh đông phương tiện.
- Trên Mac M1 Pro dùng MPS, YOLO26n-seg có tổng thời gian eval tốt hơn trong thí nghiệm này.

## File Kết Quả

- Tổng hợp: `outputs/compare/comparison_summary.csv`
- Theo lớp: `outputs/compare/comparison_per_class.csv`
- Biểu đồ: `outputs/compare/comparison_bar.png`
- YOLO26n-seg validation: `outputs/val/yolo26n-seg_coco_val2017/`
- YOLO11n-seg validation: `outputs/val/yolo11n-seg_coco_val2017/`

