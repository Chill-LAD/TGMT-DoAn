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
├── compare_models.py          # Train & so sánh 4 mô hình
├── create_invisible_dataset.py # Tạo invisible watermark dataset
├── baseline_resnet.py          # Baseline 1: ResNet18 (RGB only)
├── baseline_mobilenet.py      # Baseline 2: MobileNetV3 (RGB only)
├── model_v1.py                 # Ours v1: Hybrid CNN-Frequency (KHÔNG có SE Attention)
├── model_v2.py                 # Ours v2: Hybrid CNN-Frequency + SE Attention
├── train_v1.py                # Huấn luyện Ours v1
├── train_v2.py                # Huấn luyện Ours v2
├── evaluate.py               # Script đánh giá (Ours v2)
├── evaluate_all.py            # Đánh giá tất cả các mô hình
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
Source/data/                           # Visible (CLWD)
├── train/
│   ├── watermark/     # 49,000 ảnh có watermark
│   └── no_watermark/  # 49,000 ảnh không watermark
├── val/
│   ├── watermark/     # 10,500 ảnh
│   └── no_watermark/  # 10,500 ảnh
└── test/
    ├── watermark/     # 10,500 ảnh
    └── no_watermark/  # 10,500 ảnh

Source/data_invisible/                 # Invisible (COCO + DCT)
├── train/watermark/     # 3,500 ảnh
├── val/watermark/       # 750 ảnh
└── test/watermark/      # 750 ảnh
```

## Huấn luyện

### Train trên Combined Dataset (Visible + Invisible)

```bash
# Train Ours v1 (không có SE Attention)
python train_v1.py --epochs 30

# Train Ours v2 (có SE Attention)
python train_v2.py --epochs 30

# Train tất cả models bằng compare_models.py
python compare_models.py --epochs 30

# Train một model cụ thể
python compare_models.py --train_single ours_v1 --epochs 30
python compare_models.py --train_single ours_v2 --epochs 30
```

### Train trên Visible Only

```bash
python train_v1.py --no_merge
python train_v2.py --no_merge
```

## Đánh giá

### Evaluate All Models

```bash
# Evaluate tất cả models trên combined dataset
python evaluate_all.py

# Evaluate trên visible only
python evaluate_all.py --no_merge

# Evaluate từng model
python evaluate_all.py --model ours_v1
python evaluate_all.py --model ours_v2
```

### Evaluate bằng evaluate.py

```bash
python evaluate.py --model_path ./checkpoints/ours_v2/best_model.pth
python evaluate.py --model_path ./checkpoints/ours_v1/best_model.pth --invisible_test_dir ./data_invisible/test
```

## Phát hiện Watermark

```bash
# Phát hiện một ảnh (v1 - không SE Attention)
python main.py detect --model ./checkpoints/ours_v1_combined/best_model.pth --input ./test.jpg --model_version v1

# Phát hiện một ảnh (v2 - có SE Attention)
python main.py detect --model ./checkpoints/ours_v2_combined/best_model.pth --input ./test.jpg --model_version v2

# Phát hiện hàng loạt
python main.py detect --model ./checkpoints/ours_v2_combined/best_model.pth --input ./test_folder --output results.txt --model_version v2

# Sử dụng TTA (Test-Time Augmentation)
python main.py detect --model ./checkpoints/ours_v2_combined/best_model.pth --input ./test.jpg --tta --model_version v2
```

## Dataset Summary

| Dataset      | Source     | Type      | Watermark | No Watermark | Total   |
| ------------ | ---------- | --------- | --------- | ------------ | ------- |
| CLWD         | arXiv 2012 | Visible   | 70,000    | 70,000       | 140,000 |
| COCO + DCT   | COCO 2017  | Invisible | 5,000     | 0            | 5,000   |
| **Combined** | Both       | Both      | 75,000    | 70,000       | 145,000 |

## Command Line Arguments

| Script              | Arguments                                                    | Description            |
| ------------------- | ------------------------------------------------------------ | ---------------------- |
| `compare_models.py` | `--visible_train_dir`, `--invisible_train_dir`, `--no_merge` | Train & compare models |
| `evaluate_all.py`   | `--visible_test_dir`, `--invisible_test_dir`, `--no_merge`   | Evaluate all models    |
| `train.py`          | `--visible_train_dir`, `--invisible_train_dir`, `--no_merge` | Train hybrid model     |
| `evaluate.py`       | `--visible_test_dir`, `--invisible_test_dir`, `--no_merge`   | Evaluate hybrid model  |
| `main.py`           | Same as above + `--train`, `--eval`, `--detect`              | Unified CLI            |

## Kiến trúc mô hình

### Ours v1 (model_v1.py)

- RGB Backbone: ResNet18 (ImageNet pretrained, 512 dim output)
- Frequency Branch: FFT → Log → CNN (32→64→128) → 128 dim
- Fusion: Concat(512+128=640) → FC(640→512)
- Classification: Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.5) → FC(256→2)
- KHÔNG có SE Attention

### Ours v2 (model_v2.py)

- RGB Backbone: ResNet18 (ImageNet pretrained, 512 dim output)
- Frequency Branch: FFT → Log → CNN (32→64→128) → 128 dim
- Fusion: Concat(512+128=640) → FC(640→512)
- Attention: SE-Net (reduction=16)
- Classification: Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.5) → FC(256→2)

## Kết quả (Evaluated on Combined Dataset)

| Metric    | Baseline 1 (ResNet18) | Baseline 2 (MobileNet) | Ours v1 | Ours v2 |
| --------- | --------------------- | ---------------------- | ------- | ------- |
| Accuracy  | 84.15%                | 78.06%                 | 85.12%  | 84.99%  |
| Precision | 87.62%                | 91.08%                 | 93.64%  | 89.29%  |
| Recall    | 80.76%                | 63.83%                 | 76.42%  | 80.66%  |
| F1-score  | 84.05%                | 75.06%                 | 84.16%  | 84.76%  |
| Inference | 4.9ms                 | 7.8ms                  | 6.1ms   | 7.3ms   |

**Ours v1** đạt accuracy cao nhất (85.12%) với precision 93.64%. **Ours v2** cân bằng hơn với F1-score 84.76% và recall 80.66%.