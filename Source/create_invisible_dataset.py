"""
Script tạo invisible watermark dataset từ COCO images.
Sử dụng DCT-based embedding để nhúng watermark ẩn vào ảnh.
Đặc điểm: watermark không nhìn thấy bằng mắt thường nhưng có thể phát hiện bằng model.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import random


class InvisibleWatermarkEmbedder:
    """
    Lớp nhúng invisible watermark sử dụng DCT.
    watermark được nhúng vào miền tần số, không thay đổi đáng kể visual appearance.
    """
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def embed_dct(self, image, message, strength=10):
        """
        Nhúng watermark sử dụng DCT (Discrete Cosine Transform).
        Args:
            image: Ảnh đầu vào (H, W, 3) hoặc (H, W)
            message: Binary message bits
            strength: Cường độ watermark (ảnh hưởng đến robustness và perceptibility)
        Returns:
            Ảnh đã nhúng watermark
        """
        h, w = image.shape[:2]
        # Crop để chia hết cho 8 (yêu cầu của DCT)
        h, w = h - h % 8, w - w % 8

        if len(image.shape) == 3:
            img_float = np.float32(image[:h, :w, :])
            watermarked = np.zeros_like(img_float)

            # Xử lý từng channel
            for c in range(3):
                channel = img_float[:, :, c]
                # DCT 2D
                dct = cv2.dct(channel)

                msg_bits = message[:dct.flatten().shape[0]]
                if len(msg_bits) == 0:
                    watermarked[:, :, c] = img_float[:, :, c]
                    continue

                dct_flat = dct.flatten()
                mid = len(dct_flat) // 4

                # Nhúng vào mid-frequency
                for i in range(min(len(msg_bits), mid)):
                    freq_idx = mid + i * 4
                    if freq_idx < len(dct_flat):
                        dct_flat[freq_idx] += msg_bits[i] * strength

                dct_unflat = dct_flat.reshape(dct.shape)
                watermarked[:, :, c] = cv2.idct(dct_unflat)

            return np.clip(watermarked, 0, 255).astype(np.uint8)
        else:
            img_float = np.float32(image[:h, :w])
            dct = cv2.dct(img_float)
            dct_flat = dct.flatten()
            mid = len(dct_flat) // 4

            message_bits = np.array(message[:len(dct_flat) // 4])
            if len(message_bits) > 0:
                for i in range(len(message_bits)):
                    freq_idx = mid + i * 4
                    if freq_idx < len(dct_flat):
                        dct_flat[freq_idx] += message_bits[i] * strength

            dct_unflat = dct_flat.reshape(dct.shape)
            watermarked = cv2.idct(dct_unflat)
            return np.clip(watermarked, 0, 255).astype(np.uint8)

    def generate_random_message(self, length=128):
        """Tạo random binary message."""
        return np.random.randint(0, 2, length)

    def embed(self, image_path, strength=10):
        """
        Nhúng watermark vào ảnh từ đường dẫn.
        Args:
            image_path: Đường dẫn đến ảnh
            strength: Cường độ watermark
        Returns:
            Ảnh đã nhúng hoặc None nếu lỗi
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        message = self.generate_random_message(128)
        watermarked = self.embed_dct(image, message, strength=strength)

        return watermarked


def create_invisible_dataset(source_dir, output_dir, num_samples=5000,
                             strength=10, train_ratio=0.7, val_ratio=0.15):
    """
    Tạo invisible watermark dataset.
    Args:
        source_dir: Thư mục chứa ảnh gốc (COCO val2017)
        output_dir: Thư mục lưu dataset đã nhúng watermark
        num_samples: Số lượng ảnh cần xử lý
        strength: Cường độ watermark
        train_ratio: Tỷ lệ train
        val_ratio: Tỷ lệ validation
    """
    embedder = InvisibleWatermarkEmbedder()

    # Lấy danh sách ảnh từ COCO
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png'))]
    image_files = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"Creating invisible watermark dataset...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Total images: {len(image_files)}")
    print(f"  Watermark strength: {strength}")
    print(f"  NOTE: Only watermarked images (no no_watermark)")

    # Tạo cấu trúc thư mục
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(split_dir, "watermark"), exist_ok=True)

    # Chia dataset
    indices = list(range(len(image_files)))
    random.seed(42)
    random.shuffle(indices)

    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices
    }

    # Xử lý từng split
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split: {len(split_indices)} images...")

        for i, idx in enumerate(tqdm(split_indices, desc=split_name)):
            img_file = image_files[idx]
            src_path = os.path.join(source_dir, img_file)

            wm_dest = os.path.join(output_dir, split_name, "watermark", f"coco_{idx:06d}_{img_file}")

            watermarked = embedder.embed(src_path, strength=strength)
            if watermarked is not None:
                cv2.imwrite(wm_dest, watermarked)

    # In tổng kết
    print("\n" + "="*50)
    print("Invisible watermark dataset created!")
    print("="*50)

    for split in ["train", "val", "test"]:
        wm_path = os.path.join(output_dir, split, "watermark")
        wm_count = len(os.listdir(wm_path)) if os.path.exists(wm_path) else 0
        print(f"{split:5s}: watermark={wm_count:5d}")


def main():
    """Entry point cho command line interface."""
    import argparse
    parser = argparse.ArgumentParser(description="Create invisible watermark dataset from COCO")
    parser.add_argument("--coco_dir", type=str, default="./coco_data/val2017",
                       help="Path to COCO images")
    parser.add_argument("--output_dir", type=str, default="./data_invisible",
                       help="Output directory for invisible watermark dataset")
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Number of images to process")
    parser.add_argument("--strength", type=int, default=10,
                       help="Watermark embedding strength")

    args = parser.parse_args()

    if not os.path.exists(args.coco_dir):
        print(f"COCO directory not found: {args.coco_dir}")
        print("Please download COCO first using: python download_coco.py")
        return

    create_invisible_dataset(
        source_dir=args.coco_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        strength=args.strength
    )


if __name__ == "__main__":
    main()