import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from config import Config, AugmentationConfig


def compute_fft_spectrum(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    log_magnitude = np.log(magnitude + 1)

    min_val = log_magnitude.min()
    max_val = log_magnitude.max()
    if max_val > min_val:
        normalized = (log_magnitude - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(log_magnitude)

    h, w = normalized.shape
    target_h, target_w = Config.image_size, Config.image_size
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    cropped = normalized[top:top+target_h, left:left+target_w]

    freq_tensor = torch.from_numpy(cropped).float()
    freq_tensor = freq_tensor.unsqueeze(0)
    freq_tensor = freq_tensor.repeat(3, 1, 1)

    return freq_tensor


class WatermarkDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="train"):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.aug_config = AugmentationConfig()

        self.wm_dir = os.path.join(root_dir, "watermark")
        self.no_wm_dir = os.path.join(root_dir, "no_watermark")

        self.wm_images = []
        self.no_wm_images = []

        if os.path.exists(self.wm_dir):
            self.wm_images = [os.path.join(self.wm_dir, f) for f in os.listdir(self.wm_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if os.path.exists(self.no_wm_dir):
            self.no_wm_images = [os.path.join(self.no_wm_dir, f) for f in os.listdir(self.no_wm_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        self.images = self.wm_images + self.no_wm_images
        self.labels = [1] * len(self.wm_images) + [0] * len(self.no_wm_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            image = self._augment(image)

        image_pil = Image.fromarray(image)

        if self.transform:
            rgb_tensor = self.transform(image_pil)
        else:
            rgb_tensor = transforms.ToTensor()(image_pil)
            rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])(rgb_tensor)

        freq_tensor = compute_fft_spectrum(image)

        return {
            "rgb": rgb_tensor,
            "frequency": freq_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }

    def _augment(self, image):
        if random.random() < self.aug_config.random_flip_prob:
            image = np.fliplr(image).copy()

        if random.random() < self.aug_config.random_flip_prob:
            angle = random.uniform(-self.aug_config.random_rotation_deg,
                                 self.aug_config.random_rotation_deg)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if random.random() < self.aug_config.jpeg_prob:
            quality = random.randint(*self.aug_config.jpeg_quality)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if random.random() < self.aug_config.gaussian_noise_prob:
            sigma = self.aug_config.gaussian_noise_sigma
            noise = np.random.normal(0, sigma, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image


def get_dataloader(data_dir, batch_size=32, num_workers=4, mode="train"):
    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    dataset = WatermarkDataset(data_dir, transform=transform, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=="train"),
                           num_workers=num_workers, pin_memory=True)

    return dataloader


def create_synthetic_dataset(data_dir, num_samples=1000):
    os.makedirs(os.path.join(data_dir, "watermark"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "no_watermark"), exist_ok=True)

    print(f"Creating synthetic dataset in {data_dir}")
    print(f"  - Watermark: {num_samples} samples")
    print(f"  - No watermark: {num_samples} samples")
    print("Note: Please replace with real dataset for actual training")