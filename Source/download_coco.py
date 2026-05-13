"""
Script tải dataset COCO 2017 validation set.
Dataset này được sử dụng để tạo invisible watermark dataset.
"""
import os
import requests
import zipfile
from tqdm import tqdm


def download_file(url, destination):
    """
    Tải file từ URL với progress bar.
    Args:
        url: URL của file cần tải
        destination: Đường dẫn lưu file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(destination, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    """
    Giải nén file ZIP.
    Args:
        zip_path: Đường dẫn file ZIP
        extract_to: Thư mục giải nén
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_coco_val(output_dir="./coco_data"):
    """
    Tải và giải nén COCO 2017 validation set.
    Dataset gồm ~5000 ảnh (~1GB).
    Args:
        output_dir: Thư mục lưu dataset
    Returns:
        Đường dẫn đến thư mục chứa ảnh
    """
    os.makedirs(output_dir, exist_ok=True)

    coco_url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(output_dir, "val2017.zip")
    extract_path = output_dir

    # Kiểm tra đã tải chưa
    if os.path.exists(os.path.join(extract_path, "val2017")):
        print(f"COCO val2017 already exists at {os.path.join(extract_path, 'val2017')}")
        return os.path.join(extract_path, "val2017")

    print("Downloading COCO 2017 val set (~1GB, 5,000 images)...")
    download_file(coco_url, zip_path)

    extract_zip(zip_path, extract_path)

    return os.path.join(extract_path, "val2017")


def main():
    """Entry point cho command line interface."""
    import argparse
    parser = argparse.ArgumentParser(description="Download COCO 2017 val dataset")
    parser.add_argument("--output_dir", type=str, default="./coco_data",
                       help="Output directory for COCO dataset")
    args = parser.parse_args()

    coco_dir = download_coco_val(args.output_dir)
    print(f"\nCOCO dataset ready at: {coco_dir}")

    num_images = len([f for f in os.listdir(coco_dir) if f.endswith('.jpg')])
    print(f"Number of images: {num_images}")


if __name__ == "__main__":
    main()