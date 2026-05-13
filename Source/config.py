"""
Cấu hình cho bài toán phát hiện watermark.
Chứa các tham số huấn luyện, đường dẫn dữ liệu và cấu hình mô hình.
"""
import os
import torch

class Config:
    """Cấu hình chính cho huấn luyện và đánh giá mô hình."""
    # Đường dẫn thư mục dữ liệu
    data_dir = "./data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Tham số huấn luyện
    batch_size = 32
    num_workers = 4
    image_size = 224

    num_epochs = 30
    learning_rate = 1e-4
    weight_decay = 1e-2

    # Tham số mô hình
    num_classes = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Thư mục lưu checkpoint và log
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"

    # Các tham số huấn luyện nâng cao
    seed = 42
    use_amp = True
    gradient_clip = 1.0

    # Chế độ tiếp tục huấn luyện
    resume = None
    test_only = False

class AugmentationConfig:
    """Cấu hình các kỹ thuật tăng cường dữ liệu (data augmentation)."""
    random_flip_prob = 0.5
    random_rotation_deg = 15
    color_jitter = {"brightness": 0.2, "contrast": 0.2}
    jpeg_prob = 0.3
    jpeg_quality = (70, 90)
    gaussian_noise_prob = 0.2
    gaussian_noise_sigma = 5

class SEAttentionConfig:
    """Cấu hình cho SE (Squeeze-and-Excitation) attention module."""
    reduction = 16

class ModelConfig:
    """Cấu hình kiến trúc mô hình."""
    backbone = "resnet18"
    pretrained = True
    freeze_backbone = False
    freq_channels = 32
    dropout = 0.5