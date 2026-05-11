import os
import torch

class Config:
    data_dir = "./data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    batch_size = 32
    num_workers = 4
    image_size = 224

    num_epochs = 30
    learning_rate = 1e-4
    weight_decay = 1e-2

    num_classes = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"

    seed = 42
    use_amp = True
    gradient_clip = 1.0

    resume = None
    test_only = False

class AugmentationConfig:
    random_flip_prob = 0.5
    random_rotation_deg = 15
    color_jitter = {"brightness": 0.2, "contrast": 0.2}
    jpeg_prob = 0.3
    jpeg_quality = (70, 90)
    gaussian_noise_prob = 0.2
    gaussian_noise_sigma = 5

class SEAttentionConfig:
    reduction = 16

class ModelConfig:
    backbone = "resnet18"
    pretrained = True
    freeze_backbone = False
    freq_channels = 32
    dropout = 0.5