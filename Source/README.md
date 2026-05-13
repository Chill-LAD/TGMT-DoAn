# Watermark Detection - Source Code

## Giới thiệu

Mã nguồn cho bài toán phát hiện watermark trong ảnh số sử dụng mô hình hybrid CNN-Frequency.

## Cấu trúc thư mục

```
Source/
├── config.py          # Cấu hình hyperparameters
├── dataset.py         # Dataset class với FFT augmentation
├── divide_dataset.py  # Script chia dataset CLWD
├── model.py           # Kiến trúc mô hình (ResNet18 + FreqBranch + SE Attention)
├── train.py           # Script huấn luyện
├── evaluate.py        # Script đánh giá
├── main.py            # CLI interface
├── requirements.txt   # Python dependencies
└── README.md          # File này
```

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

File: `CLWD.rar`

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
│   └── no_watermark/  # 49,000 ảnh
├── val/
│   ├── watermark/    # 10,500 ảnh
│   └── no_watermark/  # 10,500 ảnh
└── test/
    ├── watermark/     # 10,500 ảnh
    └── no_watermark/  # 10,500 ảnh
```

## Huấn luyện

```bash
python main.py train --train_dir ./data/train --val_dir ./data/val --epochs 30
```

Các tham số huấn luyện mặc định trong `config.py`.

## Đánh giá

```bash
# Eval tất cả test images
python main.py eval --model_path ./checkpoints/best_model.pth --test_dir ./data/test

# Eval riêng từng class
python main.py eval --model_path ./checkpoints/best_model.pth --test_dir ./data/test --conditions watermark no_watermark
```

## Phát hiện watermark

Phát hiện một ảnh:

```bash
python main.py detect --model ./checkpoints/best_model.pth --input ./test/image.jpg
```

Phát hiện hàng loạt:

```bash
python main.py detect --model ./checkpoints/best_model.pth --input ./test_folder --output results.txt
```

Sử dụng Test-Time Augmentation (TTA):

```bash
python main.py detect --model ./checkpoints/best_model.pth --input ./test/image.jpg --tta
```

## Mô hình đề xuất

Kiến trúc Hybrid CNN-Frequency:

| Component        | Mô tả                                            |
| ---------------- | ------------------------------------------------ |
| RGB Backbone     | ResNet18 (ImageNet pretrained, 11.7M params)     |
| Frequency Branch | FFT → Log → Normalize → 1D CNN (128 dim)         |
| Fusion           | Concat → FC(640→512)                             |
| Attention        | SE-Net (reduction=16)                            |
| Output           | Binary classification (Watermark / No Watermark) |

## Kết quả

| Metric    | Baseline | Ours   |
| --------- | -------- | ------ |
| Accuracy  | 85%      | 90%    |
| F1-score  | 0.85     | 0.90   |
| Precision | 83%      | 88%    |
| Recall    | 87%      | 92%    |
| Inference | <50ms    | <100ms |