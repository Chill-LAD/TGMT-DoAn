# Phát hiện Watermark trong ảnh số

Mô hình hybrid kết hợp **CNN** (đặc trưng không gian) với **phân tích miền tần số** (FFT) để phát hiện watermark trong ảnh số, hỗ trợ cả **visible** và **invisible** watermarks. Kiến trúc dual-branch sử dụng ResNet18 làm RGB backbone, 1D CNN cho frequency features, và cơ chế **SE-Net Attention** để tăng cường đặc trưng.

## Cấu trúc dự án

```
./
├── Source/                 # Mã nguồn Python (models, training, evaluation)
│   ├── README.md           # Hướng dẫn cài đặt và sử dụng chi tiết
│   ├── model_v2.py         # Mô hình hybrid CNN-Frequency + SE Attention
│   ├── detect.py           # CLI phát hiện watermark (có TTA, Multi-scale)
│   └── ...
├── Docs/
│   └── BaoCao.pdf           # Báo cáo đồ án (file chính)
├── Demo/
│   └── demo.txt            # Link video demo
└── Watermarking_Detection_Proposal.pdf  # Đề xuất đồ án
```

## Tính năng chính

- **4 mô hình so sánh:** ResNet18 baseline, MobileNetV3 baseline, Ours v1 (Hybrid CNN-Freq), Ours v2 (+ SE Attention)
- **Combined dataset:** 145.000 ảnh (140.000 visible từ CLWD + 5.000 invisible từ COCO)
- **Advanced inference:** Test-Time Augmentation (TTA), Multi-scale
- **F1-score tốt nhất:** **85.39%** với cấu hình Ours v2 + TTA trên tập test 21.750 mẫu

## Quick Start

```bash
# 1. Cài đặt dependencies
cd Source
pip install -r requirements.txt

# 2. Download và chuẩn bị dữ liệu (xem chi tiết trong Source/README.md)
# 3. Huấn luyện models
python compare_models.py --epochs 30

# 4. Đánh giá
python evaluate_all.py
```

Xem chi tiết cài đặt, cấu trúc thư mục, command line arguments tại [`Source/README.md`](Source/README.md).

## Tài liệu

| Tài liệu                                                                       | Mô tả                                                                |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| [**Docs/BaoCao.pdf**](Docs/BaoCao.pdf)                                         | Báo cáo đồ án đầy đủ (kiến trúc, thực nghiệm, kết quả, so sánh SOTA) |
| [**Source/README.md**](Source/README.md)                                       | Hướng dẫn sử dụng mã nguồn chi tiết                                  |
| [**Watermarking_Detection_Proposal.pdf**](Watermarking_Detection_Proposal.pdf) | Đề xuất đồ án ban đầu                                                |
| [**Demo/demo.txt**](Demo/demo.txt)                                             | Link video demo hệ thống                                             |

## Công nghệ sử dụng

- **Python 3.8+**, **PyTorch 2.0+**, **torchvision**
- **CUDA 11.8+** (khuyến nghị GPU 8GB+ VRAM, ví dụ RTX 3060)
- **numpy, scikit-learn, Pillow, opencv-python, tqdm, tensorboard**
