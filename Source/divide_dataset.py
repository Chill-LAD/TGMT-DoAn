"""
Script chia dataset CLWD thành train/val/test splits.
Dataset CLWD chứa visible watermarks, được chia theo tỷ lệ 70/15/15.
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def setup_folders(data_root):
    """Tạo cấu trúc thư mục cho dataset: train/val/test x watermark/no_watermark."""
    for split in ["train", "val", "test"]:
        for subdir in ["watermark", "no_watermark"]:
            path = os.path.join(data_root, split, subdir)
            os.makedirs(path, exist_ok=True)
    print("Folder structure created.")


def get_image_files(directory):
    """Lấy danh sách các file ảnh trong thư mục."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if Path(f).suffix.lower() in extensions:
                files.append(os.path.join(directory, f))
    return files


def split_dataset(files, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Chia dataset thành train/val/test.
    Args:
        files: Danh sách các file
        train_ratio: Tỷ lệ train (mặc định 70%)
        val_ratio: Tỷ lệ validation (mặc định 15%)
        seed: Random seed cho reproducibility
    Returns:
        train_files, val_files, test_files
    """
    random.seed(seed)
    random.shuffle(files)

    total = len(files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]

    return train_files, val_files, test_files


def copy_files(file_list, dest_dir, prefix=""):
    """
    Copy files đến thư mục đích với tiền tố để tránh trùng lặp tên.
    Args:
        file_list: Danh sách các file nguồn
        dest_dir: Thư mục đích
        prefix: Tiền tố cho tên file (vd: wm, nwm)
    Returns:
        Số files đã copy thành công
    """
    copied = 0
    for i, src_path in enumerate(file_list):
        filename = f"{prefix}_{i:06d}_{Path(src_path).name}"
        dest_path = os.path.join(dest_dir, filename)

        try:
            shutil.copy2(src_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

    return copied


def divide_clwd_dataset(data_root, clwd_root=None):
    """
    Hàm chính để chia CLWD dataset.
    - Tìm kiếm watermark và no_watermark images từ cả train và test folders của CLWD
    - Chia thành train/val/test theo tỷ lệ 70/15/15
    Args:
        data_root: Thư mục đích để lưu dataset đã chia
        clwd_root: Thư mục gốc của CLWD dataset
    """
    if clwd_root is None:
        clwd_root = os.path.join(data_root, "CLWD", "CLWD")

    print(f"CLWD root: {clwd_root}")

    # Đường dẫn đến các thư mục trong CLWD
    watermark_source = os.path.join(clwd_root, "train", "Watermarked_image")
    nowatermark_source = os.path.join(clwd_root, "train", "Watermark_free_image")

    test_watermark = os.path.join(clwd_root, "test", "Watermarked_image")
    test_nowatermark = os.path.join(clwd_root, "test", "Watermark_free_image")

    print("\nScanning CLWD dataset...")
    print(f"  Train watermarked: {len(os.listdir(watermark_source))} images")
    print(f"  Train watermark-free: {len(os.listdir(nowatermark_source))} images")
    print(f"  Test watermarked: {len(os.listdir(test_watermark))} images")
    print(f"  Test watermark-free: {len(os.listdir(test_nowatermark))} images")

    # Gộp train và test images
    watermark_files = get_image_files(watermark_source) + get_image_files(test_watermark)
    nowatermark_files = get_image_files(nowatermark_source) + get_image_files(test_nowatermark)

    print(f"\nTotal watermark images: {len(watermark_files)}")
    print(f"Total no-watermark images: {len(nowatermark_files)}")

    # Chia dataset
    wm_train, wm_val, wm_test = split_dataset(watermark_files.copy())
    nwm_train, nwm_val, nwm_test = split_dataset(nowatermark_files.copy())

    print(f"\nSplit summary:")
    print(f"  Train: {len(wm_train)} watermark, {len(nwm_train)} no_watermark")
    print(f"  Val:   {len(wm_val)} watermark, {len(nwm_val)} no_watermark")
    print(f"  Test:  {len(wm_test)} watermark, {len(nwm_test)} no_watermark")

    setup_folders(data_root)

    # Copy watermark images
    print("\nCopying watermark images...")
    print("  Train:")
    copy_files(wm_train, os.path.join(data_root, "train", "watermark"), "wm")
    print("  Val:")
    copy_files(wm_val, os.path.join(data_root, "val", "watermark"), "wm")
    print("  Test:")
    copy_files(wm_test, os.path.join(data_root, "test", "watermark"), "wm")

    # Copy no_watermark images
    print("\nCopying no_watermark images...")
    print("  Train:")
    copy_files(nwm_train, os.path.join(data_root, "train", "no_watermark"), "nwm")
    print("  Val:")
    copy_files(nwm_val, os.path.join(data_root, "val", "no_watermark"), "nwm")
    print("  Test:")
    copy_files(nwm_test, os.path.join(data_root, "test", "no_watermark"), "nwm")

    print("\n" + "="*50)
    print("Dataset division complete!")
    print("="*50)

    # In tổng kết
    print("\nFinal dataset summary:")
    for split in ["train", "val", "test"]:
        wm_path = os.path.join(data_root, split, "watermark")
        nwm_path = os.path.join(data_root, split, "no_watermark")

        wm_count = len(os.listdir(wm_path)) if os.path.exists(wm_path) else 0
        nwm_count = len(os.listdir(nwm_path)) if os.path.exists(nwm_path) else 0

        total = wm_count + nwm_count
        ratio = (wm_count / total * 100) if total > 0 else 0

        print(f"  {split:5s}: watermark={wm_count:5d}, no_watermark={nwm_count:5d}, total={total:5d}, wm_ratio={ratio:.1f}%")


def main():
    """Entry point cho command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Divide CLWD dataset into train/val/test splits")
    parser.add_argument("--data_root", type=str, default=None,
                       help="Root directory for dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_root is None:
        args.data_root = os.path.join(script_dir, "data")

    print(f"Data root: {os.path.abspath(args.data_root)}")

    divide_clwd_dataset(args.data_root)


if __name__ == "__main__":
    main()