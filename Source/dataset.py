"""
Dataset và các lớp tiền xử lý cho bài toán phát hiện watermark.
Bao gồm: RGBPreprocessor, FFTPreprocessor, WatermarkPreprocessor, WatermarkDataset.
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import Config


# RGB PREPROCESSOR - Tiền xử lý ảnh RGB

class RGBPreprocessor:
    """
    Tiền xử lý ảnh RGB: resize, normalize theo ImageNet.
    Áp dụng augmentation khi huấn luyện (flip, rotation, color jitter).
    """
    def __init__(self, image_size=224, train=True):
        self.image_size = image_size
        self.train = train

        if self.train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image):
        return self.transform(image)


# FFT PREPROCESSOR - Tiền xử lý miền tần số

class FFTPreprocessor:
    """
    Chuyển đổi ảnh sang miền tần số sử dụng FFT.
    Trích xuất đặc trưng tần số để phát hiện invisible watermark.
    """
    def __init__(self, image_size=224):
        self.image_size = image_size

    def compute_fft(self, image):
        """Tính FFT 2D và lấy log magnitude spectrum."""
        # Chuyển sang grayscale nếu là ảnh màu
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        # Tính FFT 2D
        fft = np.fft.fft2(gray)
        # Dịch zero-frequency component ra giữa
        fft_shift = np.fft.fftshift(fft)
        # Lấy magnitude
        magnitude = np.abs(fft_shift)
        # Log transform để compress dynamic range
        log_magnitude = np.log(magnitude + 1)
        return log_magnitude

    def normalize(self, image):
        """Chuẩn hóa giá trị về [0, 1]."""
        min_val, max_val = image.min(), image.max()
        return (image - min_val) / (max_val - min_val + 1e-8)

    def __call__(self, image):
        """Xử lý ảnh: FFT -> normalize -> resize -> convert to tensor."""
        image = np.array(image)
        fft_image = self.compute_fft(image)
        fft_image = self.normalize(fft_image)
        fft_image = cv2.resize(fft_image, (self.image_size, self.image_size))
        fft_image = fft_image.astype(np.float32)
        # Expand thành 3 channels để tương thích với ResNet
        fft_image = np.expand_dims(fft_image, axis=0)
        fft_image = np.repeat(fft_image, 3, axis=0)
        return torch.from_numpy(fft_image)


# AUGMENTATIONS - Các kỹ thuật tăng cường dữ liệu

class JPEGCompression:
    """
    Augmentation: Mô phỏng nén JPEG.
    Giúp mô hình robust với ảnh bị nén.
    """
    def __init__(self, quality_range=(70, 90)):
        self.quality_range = quality_range

    def __call__(self, image):
        image = np.array(image)
        quality = np.random.randint(*self.quality_range)
        # Chuyển RGB sang BGR cho OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Nén và giải nén
        _, encoded = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


class GaussianNoise:
    """
    Augmentation: Thêm nhiễu Gaussian.
    Giúp mô hình robust với nhiễu.
    """
    def __init__(self, sigma_range=(5, 15)):
        self.sigma_range = sigma_range

    def __call__(self, image):
        image = np.array(image)
        sigma = np.random.uniform(*self.sigma_range)
        noise = np.random.normal(0, sigma, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


# MAIN PREPROCESSOR - Tiền xử lý chính

class WatermarkPreprocessor:
    """
    Tiền xử lý chính: kết hợp RGB + FFT preprocessors và augmentation.
    Trả về 2 tensor: rgb_tensor và freq_tensor.
    """
    def __init__(self, image_size=224, train=True,
                 use_jpeg_aug=False, use_noise_aug=False):
        self.rgb_processor = RGBPreprocessor(image_size=image_size, train=train)
        self.fft_processor = FFTPreprocessor(image_size=image_size)
        self.train = train

        self.augmentations = []
        if use_jpeg_aug:
            self.augmentations.append(JPEGCompression())
        if use_noise_aug:
            self.augmentations.append(GaussianNoise())

    def __call__(self, image):
        # Áp dụng augmentation với xác suất 30%
        if self.train and self.augmentations and np.random.random() < 0.3:
            for aug in self.augmentations:
                image = aug(image)

        rgb_tensor = self.rgb_processor(image)
        freq_tensor = self.fft_processor(image)

        return rgb_tensor, freq_tensor


# DATASET - Lớp Dataset cho PyTorch

class WatermarkDataset(Dataset):
    """
    Dataset cho bài toán phát hiện watermark.
    Đọc ảnh từ thư mục với cấu trúc: root/watermark/* và root/no_watermark/*.
    """
    def __init__(self, root_dir, image_size=224, train=True,
                 use_jpeg_aug=False, use_noise_aug=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.train = train

        self.class_to_idx = {"no_watermark": 0, "watermark": 1}
        self.samples = []

        self.preprocessor = WatermarkPreprocessor(
            image_size=image_size,
            train=train,
            use_jpeg_aug=use_jpeg_aug,
            use_noise_aug=use_noise_aug
        )

        self.load_dataset()

    def load_dataset(self):
        """Quét thư mục và thu thập danh sách ảnh."""
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for image_name in os.listdir(class_dir):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(class_dir, image_name)
                    self.samples.append((image_path, label))

        np.random.seed(42)
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Trả về một sample: rgb tensor, frequency tensor, và label."""
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        rgb_tensor, freq_tensor = self.preprocessor(image)

        return {
            "rgb": rgb_tensor,
            "frequency": freq_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }


# DATALOADER UTILITY - Hàm tiện ích tạo DataLoader

def get_dataloader(data_dir, batch_size=32, num_workers=4, mode="train",
                   use_jpeg_aug=False, use_noise_aug=False):
    """
    Tạo DataLoader cho watermark dataset.
    Args:
        data_dir: Đường dẫn thư mục dữ liệu
        batch_size: Kích thước batch
        num_workers: Số workers cho DataLoader
        mode: "train", "val", hoặc "test"
        use_jpeg_aug: Có sử dụng JPEG augmentation không
        use_noise_aug: Có sử dụng noise augmentation không
    """
    dataset = WatermarkDataset(
        root_dir=data_dir,
        image_size=Config.image_size,
        train=(mode == "train"),
        use_jpeg_aug=use_jpeg_aug,
        use_noise_aug=use_noise_aug
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# SYNTHETIC DATASET - Tạo dataset giả lập

def create_synthetic_dataset(data_dir, num_samples=1000):
    """
    Tạo cấu trúc thư mục cho synthetic dataset.
    Chỉ tạo thư mục, cần thay thế bằng dataset thực để huấn luyện.
    """
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(data_dir, split, "watermark"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, "no_watermark"), exist_ok=True)

    print(f"Creating synthetic dataset in {data_dir}")
    print(f"  - Watermark: {num_samples} samples")
    print(f"  - No watermark: {num_samples} samples")
    print("Note: Please replace with real dataset for actual training")


# LEGACY COMPATIBILITY - Hàm tương thích

def compute_fft_spectrum(image):
    """
    Tính FFT spectrum cho ảnh đầu vào.
    Hàm standalone để sử dụng trong inference.
    Args:
        image: Ảnh numpy array (H, W, 3) hoặc (H, W)
    Returns:
        Tensor shape (3, image_size, image_size)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log(magnitude + 1)

    min_val, max_val = log_magnitude.min(), log_magnitude.max()
    normalized = (log_magnitude - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(log_magnitude)

    h, w = normalized.shape
    target_h, target_w = Config.image_size, Config.image_size
    top, left = (h - target_h) // 2, (w - target_w) // 2
    cropped = normalized[top:top+target_h, left:left+target_w]

    freq_tensor = torch.from_numpy(cropped).float().unsqueeze(0).repeat(3, 1, 1)
    return freq_tensor