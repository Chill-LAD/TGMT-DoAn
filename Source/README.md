# Watermark Detection - Source Code

## Giới thiệu

Mã nguồn cho bài toán phát hiện watermark trong ảnh số sử dụng mô hình hybrid CNN-Frequency.

## Cấu trúc thư mục

```
Source/
├── config.py             # Cấu hình hyperparameters
├── dataset.py            # Dataset class với FFT augmentation
├── divide_dataset.py     # Script chia dataset CLWD
├── baseline_resnet.py    # Baseline 1: ResNet18 (RGB only)
├── baseline_mobilenet.py # Baseline 2: MobileNetV3 (RGB only)
├── model.py              # Ours v2: Hybrid CNN-Frequency + SE Attention
├── compare_models.py     # Train & so sánh 4 mô hình
├── evaluate_all.py       # Đánh giá tất cả các mô hình
├── train.py              # Script huấn luyện (hybrid model)
├── evaluate.py           # Script đánh giá (hybrid model)
├── main.py               # CLI interface
├── requirements.txt      # Python dependencies
└── README.md             # File này
```

## Mô hình

| Mô hình        | Mô tả                               | Params |
| -------------- | ----------------------------------- | ------ |
| **Baseline 1** | ResNet18 (RGB only)                 | 11.7M  |
| **Baseline 2** | MobileNetV3 (RGB only)              | 3.5M   |
| **Ours v1**    | ResNet18 + Frequency branch         | ~12M   |
| **Ours v2**    | ResNet18 + Frequency + SE Attention | ~12M   |

## Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (nếu sử dụng GPU)
- GPU VRAM: 8GB+ (RTX 3060 recommended)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

### 1. Download CLWD Dataset

Download từ Google Drive: https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij

File: `CLWD.rar` (~3.5GB)

### 2. Extract

Extract vào `Source/data/CLWD/CLWD/` với cấu trúc:

```
CLWD/
├── train/
│   ├── Watermarked_image/    # 60,000 ảnh
│   └── Watermark_free_image/ # 60,000 ảnh
└── test/
    ├── Watermarked_image/     # 10,000 ảnh
    └── Watermark_free_image/  # 10,000 ảnh
```

### 3. Chia dataset

```bash
python divide_dataset.py
```

Dataset structure sau khi chia:

```
data/
├── train/
│   ├── watermark/     # 49,000 ảnh
│   └── no_watermark/ # 49,000 ảnh
├── val/
│   ├── watermark/    # 10,500 ảnh
│   └── no_watermark/ # 10,500 ảnh
└── test/
    ├── watermark/    # 10,500 ảnh
    └── no_watermark/ # 10,500 ảnh
```

## Huấn luyện

### Train từng mô hình

```bash
# Train baseline ResNet18
python compare_models.py --train_single resnet18 --train_dir ./data/train --val_dir ./data/val --epochs 30

# Train baseline MobileNetV3
python compare_models.py --train_single mobilenet --train_dir ./data/train --val_dir ./data/val --epochs 30

# Train hybrid model (Ours v2)
python compare_models.py --train_single hybrid --train_dir ./data/train --val_dir ./data/val --epochs 30
```

### Train tất cả và so sánh

```bash
python compare_models.py --train_dir ./data/train --val_dir ./data/val --epochs 30
```

Checkpoints sẽ được lưu trong:

```
checkpoints/
├── resnet18/best_model.pth
├── mobilenet/best_model.pth
└── hybrid/best_model.pth
```

## Đánh giá

### Đánh giá tất cả các mô hình

```bash
python evaluate_all.py --checkpoint_dir ./checkpoints --test_dir ./data/test
```

### Đánh giá từng mô hình

```bash
python evaluate_all.py --checkpoint_dir ./checkpoints --test_dir ./data/test --model resnet18
python evaluate_all.py --checkpoint_dir ./checkpoints --test_dir ./data/test --model mobilenet
python evaluate_all.py --checkpoint_dir ./checkpoints --test_dir ./data/test --model hybrid
```

### Đánh giá hybrid model (CLI)

```bash
python main.py eval --model_path ./checkpoints/hybrid/best_model.pth --test_dir ./data/test
```

## Phát hiện watermark

```bash
# Phát hiện một ảnh
python main.py detect --model ./checkpoints/hybrid/best_model.pth --input ./test.jpg

# Phát hiện hàng loạt
python main.py detect --model ./checkpoints/hybrid/best_model.pth --input ./test_folder --output results.txt

# Sử dụng TTA
python main.py detect --model ./checkpoints/hybrid/best_model.pth --input ./test.jpg --tta
```

## Kiến trúc mô hình đề xuất (Ours v2)

| Component        | Mô tả                                            |
| ---------------- | ------------------------------------------------ |
| RGB Backbone     | ResNet18 (ImageNet pretrained, 11.7M params)     |
| Frequency Branch | FFT → Log → Normalize → 1D CNN (128 dim)         |
| Fusion           | Concat → FC(640→512)                             |
| Attention        | SE-Net (reduction=16)                            |
| Classification   | FC(512→256) + Dropout(0.5) → FC(256→2)           |
| Output           | Binary classification (Watermark / No Watermark) |

## Kết quả

| Metric    | Baseline 1 | Baseline 2 | Ours v1 | Ours v2 |
| --------- | ---------- | ---------- | ------- | ------- |
| Accuracy  | 85%        | 82%        | 88%     | 90%     |
| F1-score  | 0.85       | 0.82       | 0.88    | 0.90    |
| Precision | 83%        | 80%        | 87%     | 88%     |
| Recall    | 87%        | 84%        | 89%     | 92%     |
| Inference | ~40ms      | ~20ms      | ~80ms   | ~90ms   |

# 