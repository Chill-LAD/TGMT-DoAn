# Watermark Detection - Source Code

## Giới thiệu

Mã nguồn cho bài toán phát hiện watermark trong ảnh số sử dụng mô hình hybrid CNN-Frequency. Hỗ trợ cả visible và invisible watermarks.

## Cấu trúc thư mục

```
Source/
├── config.py                   # Cấu hình hyperparameters
├── dataset.py                  # Dataset class (visible watermarks)
├── dual_dataset.py             # Dataset class (visible + invisible)
├── divide_dataset.py           # Chia dataset CLWD
├── download_coco.py           # Download COCO 2017 val
├── create_invisible_dataset.py # Tạo invisible watermark dataset
├── baseline_resnet.py          # Baseline 1: ResNet18 (RGB only)
├── baseline_mobilenet.py      # Baseline 2: MobileNetV3 (RGB only)
├── model.py                    # Ours v2: Hybrid CNN-Frequency + SE Attention
├── compare_models.py           # Train & so sánh 4 mô hình
├── evaluate_all.py             # Đánh giá tất cả các mô hình
├── train.py                    # Script huấn luyện (hybrid model)
├── evaluate.py                 # Script đánh giá (hybrid model)
├── main.py                     # CLI interface
├── requirements.txt            # Python dependencies
└── README.md                   # File này
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

### Dataset 1: Visible Watermark (CLWD)

```bash
# 1. Download CLWD từ Google Drive: https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij
# File: CLWD.rar (~3.5GB)

# 2. Extract vào Source/data/CLWD/CLWD/

# 3. Chia dataset
python divide_dataset.py
```

### Dataset 2: Invisible Watermark (COCO + Synthetic)

```bash
# 1. Download COCO 2017 val
python download_coco.py --output_dir ./coco_data

# 2. Tạo invisible watermark dataset (5,000 images, không có no_watermark)
python create_invisible_dataset.py --coco_dir ./coco_data/val2017 --output_dir ./data_invisible --num_samples 5000
```

### Dataset Structure

```
data/                           # Visible (CLWD)
├── train/
│   ├── watermark/     # ~49,000 ảnh có watermark
│   └── no_watermark/  # ~49,000 ảnh không watermark
├── val/
│   ├── watermark/     # ~10,500 ảnh
│   └── no_watermark/  # ~10,500 ảnh
└── test/
    ├── watermark/     # ~10,500 ảnh
    └── no_watermark/  # ~10,500 ảnh

data_invisible/                 # Invisible (COCO + DCT)
├── train/watermark/     # ~3,500 ảnh
├── val/watermark/       # ~750 ảnh
└── test/watermark/      # ~750 ảnh
```

## Huấn luyện

### Train trên Combined Dataset (Visible + Invisible)

```bash
# Train hybrid model (default: merge cả visible + invisible)
python compare_models.py --train_single hybrid --epochs 30

# Train tất cả models
python compare_models.py --epochs 30

# Train với batch size lớn hơn
python compare_models.py --train_single hybrid --epochs 30 --batch_size 64
```

### Train trên Visible Only

```bash
python compare_models.py --train_single resnet18 --no_merge
```

### Train bằng train.py (hybrid only)

```bash
# Combined dataset
python train.py --epochs 30

# Visible only
python train.py --no_merge
```

### Train bằng main.py (hybrid only)

```bash
python main.py train --epochs 30
python main.py train --visible_train_dir ./data/train --visible_val_dir ./data/val --no_merge
```

## Đánh giá

### Evaluate All Models

```bash
# Evaluate tất cả models trên combined dataset
python evaluate_all.py

# Evaluate trên visible only
python evaluate_all.py --no_merge

# Evaluate từng model
python evaluate_all.py --model hybrid
```

### Evaluate bằng evaluate.py

```bash
python evaluate.py --model_path ./checkpoints/best_model.pth
python evaluate.py --model_path ./checkpoints/best_model.pth --invisible_test_dir ./data_invisible/test
```

### Evaluate bằng main.py

```bash
python main.py eval --model_path ./checkpoints/best_model.pth
```

## Phát hiện Watermark

```bash
# Phát hiện một ảnh
python main.py detect --model ./checkpoints/best_model.pth --input ./test.jpg

# Phát hiện hàng loạt
python main.py detect --model ./checkpoints/best_model.pth --input ./test_folder --output results.txt

# Sử dụng TTA (Test-Time Augmentation)
python main.py detect --model ./checkpoints/best_model.pth --input ./test.jpg --tta
```

## Dataset Summary

| Dataset        | Source           | Type      | Watermark | No Watermark | Total   |
| -------------- | ---------------- | --------- | --------- | ------------ | ------- |
| CLWD           | arXiv 2012.07616 | Visible   | 70,000    | 70,000       | 140,000 |
| COCO val + DCT | COCO 2017        | Invisible | 5,000     | 0            | 5,000   |
| **Combined**   | Both             | Both      | 75,000    | 70,000       | 145,000 |

## Command Line Arguments

| Script              | Arguments                                                    | Description            |
| ------------------- | ------------------------------------------------------------ | ---------------------- |
| `compare_models.py` | `--visible_train_dir`, `--invisible_train_dir`, `--no_merge` | Train & compare models |
| `evaluate_all.py`   | `--visible_test_dir`, `--invisible_test_dir`, `--no_merge`   | Evaluate all models    |
| `train.py`          | `--visible_train_dir`, `--invisible_train_dir`, `--no_merge` | Train hybrid model     |
| `evaluate.py`       | `--visible_test_dir`, `--invisible_test_dir`, `--no_merge`   | Evaluate hybrid model  |
| `main.py`           | Same as above + `--train`, `--eval`, `--detect`              | Unified CLI            |

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