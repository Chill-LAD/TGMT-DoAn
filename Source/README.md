# Watermark Detection - Source Code

## Giới thiệu

Đây là mã nguồn cho bài toán phát hiện watermark trong ảnh số sử dụng mô hình hybrid CNN-Frequency.

## Cấu trúc thư mục

```
Source/
├── config.py          # Cấu hình hyperparameters
├── dataset.py         # Dataset class với FFT augmentation
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

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Tổ chức dữ liệu theo cấu trúc:

```
data/
├── train/
│   ├── watermark/     # Ảnh có watermark
│   └── no_watermark/ # Ảnh không có watermark
├── val/
│   ├── watermark/
│   └── no_watermark/
└── test/
    ├── watermark/
    ├── no_watermark/
    ├── clean/        # Test không distortion
    ├── jpeg/        # Test JPEG compression
    ├── resize/      # Test resize
    └── noise/       # Test noise
```

## Huấn luyện

```bash
python main.py train --train_dir ./data/train --val_dir ./data/val --epochs 30
```

Các tham số huấn luyện mặc định trong `config.py`.

## Đánh giá

```bash
python main.py eval --model_path ./checkpoints/best_model.pth --test_dir ./data/test
```

Đánh giá theo điều kiện:

```bash
python main.py eval --model_path ./checkpoints/best_model.pth --test_dir ./data/test --conditions clean jpeg resize noise
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

Kiến trúc mô hình Hybrid CNN-Frequency:

- RGB Backbone: ResNet18 (ImageNet pretrained)
- Frequency Branch: 1D CNN với FFT preprocessing
- SE-Net Attention
- Binary classification (Watermark / No Watermark)

## Kết quả

| Metric    | Giá trị |
| --------- | ------- |
| Accuracy  | 90%     |
| F1-score  | 0.90    |
| Precision | 88%     |
| Recall    | 92%     |
| Inference | <100ms  |

# 