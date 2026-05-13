"""
Dataset kết hợp cho cả visible và invisible watermarks.
Hỗ trợ merge dataset từ nhiều nguồn: CLWD (visible) và COCO+DCT (invisible).
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class DualWatermarkDataset(Dataset):
    """
    Dataset kết hợp visible và invisible watermarks.
    - Visible: từ CLWD dataset (label=1 cho watermark, label=0 cho no_watermark)
    - Invisible: từ COCO + DCT embedding (chỉ có watermark, label=1)
    """
    def __init__(self, visible_dir=None, invisible_dir=None,
                 image_size=224, train=True, merge=True):
        self.image_size = image_size
        self.train = train
        self.samples = []

        # Merge cả visible và invisible datasets
        if merge:
            if visible_dir and os.path.exists(visible_dir):
                self.samples.extend(self._load_split(visible_dir, "visible", train))

            if invisible_dir and os.path.exists(invisible_dir):
                self.samples.extend(self._load_split_invisible(invisible_dir, train))
        else:
            # Chỉ sử dụng một dataset
            if visible_dir and os.path.exists(visible_dir):
                self.samples.extend(self._load_split(visible_dir, "visible", train))
            elif invisible_dir and os.path.exists(invisible_dir):
                self.samples.extend(self._load_split_invisible(invisible_dir, train))

        random.seed(42)
        random.shuffle(self.samples)

        self.transform = self._get_transform()

    def _load_split(self, data_dir, dataset_type, mode):
        """Load visible watermark dataset từ thư mục."""
        samples = []
        subdirs_to_check = ["watermark", "no_watermark"]

        split = "train" if mode else "test"
        search_dirs = []
        # Tìm kiếm trong cả các thư mục con và thư mục gốc
        for subdir in subdirs_to_check:
            for s in ["train", "val", "test"]:
                search_dirs.append((os.path.join(data_dir, s, subdir), 1 if subdir == "watermark" else 0))
            search_dirs.append((os.path.join(data_dir, subdir), 1 if subdir == "watermark" else 0))

        for full_dir, label in search_dirs:
            if not os.path.exists(full_dir):
                continue

            for img_name in os.listdir(full_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(full_dir, img_name)
                    samples.append({
                        "path": img_path,
                        "label": label,
                        "type": dataset_type
                    })

        return samples

    def _load_split_invisible(self, data_dir, mode):
        """Load invisible watermark dataset (chỉ có thư mục watermark)."""
        samples = []
        watermark_dir = os.path.join(data_dir, "watermark")

        if os.path.exists(watermark_dir):
            for img_name in os.listdir(watermark_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(watermark_dir, img_name)
                    samples.append({
                        "path": img_path,
                        "label": 1,
                        "type": "invisible"
                    })

        return samples

    def _get_transform(self):
        """Tạo transform pipeline cho training/validation/testing."""
        if self.train:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def _compute_fft(self, image):
        """Tính FFT spectrum cho ảnh đầu vào."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, (self.image_size, self.image_size))
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        log_magnitude = np.log(magnitude + 1)

        normalized = log_magnitude
        if normalized.max() > normalized.min():
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)

        freq_tensor = torch.from_numpy(normalized).float().unsqueeze(0).contiguous()
        freq_tensor = freq_tensor.repeat(3, 1, 1)

        return freq_tensor

    def _augment(self, image):
        """Áp dụng các kỹ thuật augmentation."""
        # Random horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()

        # JPEG compression simulation
        if random.random() < 0.3:
            quality = random.randint(70, 90)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Gaussian noise
        if random.random() < 0.2:
            sigma = random.uniform(5, 15)
            noise = np.random.normal(0, sigma, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Trả về một sample với RGB tensor, frequency tensor và label."""
        sample = self.samples[idx]
        img_path = sample["path"]
        label = sample["label"]

        # Đọc ảnh bằng OpenCV (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            # Ảnh lỗi: trả về ảnh đen
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Áp dụng augmentation khi training
        if self.train:
            image = self._augment(image)

        image_pil = Image.fromarray(image)
        rgb_tensor = self.transform(image_pil)

        # Tính FFT spectrum
        freq_tensor = self._compute_fft(np.array(image_pil))

        return {
            "rgb": rgb_tensor,
            "frequency": freq_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }


def get_dual_dataloader(visible_dir=None, invisible_dir=None,
                       batch_size=32, num_workers=4, mode="train",
                       merge=True):
    """
    Tạo DataLoader cho DualWatermarkDataset.
    Args:
        visible_dir: Đường dẫn dataset visible watermark
        invisible_dir: Đường dẫn dataset invisible watermark
        batch_size: Kích thước batch
        num_workers: Số workers
        mode: "train", "val", hoặc "test"
        merge: Có merge cả 2 datasets không
    """
    dataset = DualWatermarkDataset(
        visible_dir=visible_dir,
        invisible_dir=invisible_dir,
        train=(mode == "train"),
        merge=merge
    )

    if len(dataset) == 0:
        print(f"Warning: No samples found in dataset!")
        print(f"  Visible dir: {visible_dir}")
        print(f"  Invisible dir: {invisible_dir}")
        print(f"  Merge: {merge}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def get_dataset_stats(visible_dir=None, invisible_dir=None, merge=True):
    """
    Thống kê số lượng mẫu trong dataset.
    Returns:
        Dict chứa số lượng watermark/no_watermark cho visible/invisible.
    """
    stats = {
        "visible_watermark": 0,
        "visible_no_watermark": 0,
        "invisible_watermark": 0,
        "invisible_no_watermark": 0
    }

    if visible_dir and os.path.exists(visible_dir):
        for split in ["train", "val", "test"]:
            for subdir, key in [("watermark", "visible_watermark"), ("no_watermark", "visible_no_watermark")]:
                full_dir = os.path.join(visible_dir, split, subdir)
                if os.path.exists(full_dir):
                    count = len([f for f in os.listdir(full_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    stats[key] += count

    if invisible_dir and os.path.exists(invisible_dir):
        for split in ["train", "val", "test"]:
            full_dir = os.path.join(invisible_dir, split, "watermark")
            if os.path.exists(full_dir):
                count = len([f for f in os.listdir(full_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                stats["invisible_watermark"] += count

    return stats