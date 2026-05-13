import os
import shutil
import zipfile
import random
from tqdm import tqdm


def setup_folders(data_root):
    for split in ["train", "val", "test"]:
        for subdir in ["watermark", "no_watermark"]:
            path = os.path.join(data_root, split, subdir)
            os.makedirs(path, exist_ok=True)
    print("Folder structure created.")


def detect_and_fix_archive(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(4)
    
    if header[:4] == b'Rar!':
        print(f"Detected RAR file (misnamed as {os.path.basename(filepath)})")
        new_path = filepath.replace('.zip', '.rar')
        os.rename(filepath, new_path)
        return new_path
    elif header[:2] == b'PK':
        print(f"Detected ZIP file")
        return filepath
    else:
        print(f"Warning: Unknown archive format. Header: {header}")
        return filepath


def download_and_extract_clwd(data_root):
    import gdown

    clwd_zip_path = os.path.join(data_root, "CLWD.zip")
    clwd_extract_path = os.path.join(data_root, "CLWD")

    if os.path.exists(clwd_extract_path):
        print(f"CLWD already extracted at {clwd_extract_path}")
        return clwd_extract_path

    if not os.path.exists(clwd_zip_path):
        print("Downloading CLWD from Google Drive...")
        gdrive_url = "https://drive.google.com/uc?id=17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij"
        gdown.download(gdrive_url, clwd_zip_path, quiet=False)
        print(f"Downloaded to {clwd_zip_path}")
    else:
        print(f"Found existing {clwd_zip_path}")

    archive_path = detect_and_fix_archive(clwd_zip_path)

    print("Extracting CLWD...")
    
    if archive_path.endswith('.rar'):
        try:
            import unrar
            unrar.unrar.extract(archive_path, data_root)
        except ImportError:
            import subprocess
            rar_exe = "C:\\Program Files\\7-Zip\\7z.exe"
            if os.path.exists(rar_exe):
                subprocess.run([rar_exe, 'x', archive_path, f'-o{data_root}', '-y'])
            else:
                print("unrar not installed. Trying to extract with pyunpack...")
                import subprocess
                subprocess.run(['python', '-m', 'pip', 'install', 'pyunpack', 'patool'])
                from pyunpack import Archive
                Archive(archive_path).extractall(data_root)
    else:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)

    return clwd_extract_path


def prepare_clwd_dataset(data_root, clwd_path):
    print("\nPreparing CLWD dataset...")

    images_dir = os.path.join(clwd_path, "images")
    watermarked_dir = os.path.join(clwd_path, "watermarked")

    if not os.path.exists(images_dir):
        print(f"Warning: {images_dir} not found")
        subdirs = os.listdir(clwd_path)
        print(f"Available folders: {subdirs}")

    watermark_images = []
    no_watermark_images = []

    if os.path.exists(watermarked_dir):
        wm_files = [f for f in os.listdir(watermarked_dir) if f.lower().endswith(('.jpg', '.png'))]
        watermark_images = [os.path.join(watermarked_dir, f) for f in wm_files]
        print(f"Found {len(watermark_images)} watermarked images")

    if os.path.exists(images_dir):
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
        no_watermark_images = [os.path.join(images_dir, f) for f in img_files]
        print(f"Found {len(no_watermark_images)} original images (no watermark)")

    all_images = watermark_images + no_watermark_images
    random.seed(42)
    random.shuffle(all_images)

    total = len(all_images)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_indices = all_images[:train_size]
    val_indices = all_images[train_size:train_size+val_size]
    test_indices = all_images[train_size+val_size:]

    print(f"\nSplit: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    for idx, img_path in enumerate(tqdm(all_images, desc="Copying images")):
        is_watermarked = img_path in watermark_images
        subdir = "watermark" if is_watermarked else "no_watermark"

        if idx < train_size:
            split = "train"
        elif idx < train_size + val_size:
            split = "val"
        else:
            split = "test"

        filename = f"clwd_{idx:06d}_{os.path.basename(img_path)}"
        dest = os.path.join(data_root, split, subdir, filename)

        try:
            shutil.copy2(img_path, dest)
        except Exception as e:
            print(f"Error copying {img_path}: {e}")

    print("\n[CLWD Dataset] Done!")


def prepare_pita_dataset(data_root):
    from datasets import load_dataset

    print("\nPreparing PITA dataset...")
    dataset = load_dataset("bastienp/visible-watermark-pita", trust_remote_code=True)

    split = dataset["train"]
    total = len(split)
    indices = list(range(total))
    random.seed(42)
    random.shuffle(indices)

    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    splits = {
        "train": indices[:train_size],
        "val": indices[train_size:train_size+val_size],
        "test": indices[train_size+val_size:]
    }

    for split_name, split_indices in splits.items():
        print(f"Processing {split_name} split: {len(split_indices)} images...")

        for i, idx in enumerate(tqdm(split_indices, desc=split_name)):
            item = split[idx]
            image = item["image"]

            filename = f"pita_{idx:06d}.png"
            dest = os.path.join(data_root, split_name, "watermark", filename)

            try:
                image.save(dest)
            except Exception as e:
                print(f"Error saving {filename}: {e}")

    print("\n[PITA Dataset] Done!")


def create_no_watermark_from_coco(data_root, num_samples=5000):
    print("\nCreating no_watermark subset from COCO...")

    coco_dir = os.path.join(data_root, "COCO")
    coco_images_dir = os.path.join(coco_dir, "train2017")

    if not os.path.exists(coco_images_dir):
        print(f"COCO not found at {coco_images_dir}")
        print("Please download from: http://images.cocodataset.org/zips/train2017.zip")
        return

    images = [f for f in os.listdir(coco_images_dir) if f.endswith(('.jpg', '.png'))]
    sampled = random.sample(images, min(num_samples, len(images)))

    train_size = int(0.7 * len(sampled))
    val_size = int(0.15 * len(sampled))

    for i, img_file in enumerate(tqdm(sampled, desc="Copying COCO images")):
        src = os.path.join(coco_images_dir, img_file)

        if i < train_size:
            split = "train"
        elif i < train_size + val_size:
            split = "val"
        else:
            split = "test"

        filename = f"coco_{img_file}"
        dest = os.path.join(data_root, split, "no_watermark", filename)

        shutil.copy2(src, dest)

    print(f"Added {len(sampled)} no_watermark images from COCO")


def verify_dataset(data_root):
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)

    total_wm = 0
    total_no_wm = 0

    for split in ["train", "val", "test"]:
        wm_path = os.path.join(data_root, split, "watermark")
        no_wm_path = os.path.join(data_root, split, "no_watermark")

        wm_count = len(os.listdir(wm_path)) if os.path.exists(wm_path) else 0
        no_wm_count = len(os.listdir(no_wm_path)) if os.path.exists(no_wm_path) else 0

        total_wm += wm_count
        total_no_wm += no_wm_count

        ratio = wm_count / (wm_count + no_wm_count) * 100 if (wm_count + no_wm_count) > 0 else 0

        print(f"{split:6s}: watermark={wm_count:5d}, no_watermark={no_wm_count:5d}, ratio={ratio:.1f}%")

    print("-"*50)
    print(f"{'Total':6s}: watermark={total_wm:5d}, no_watermark={total_no_wm:5d}")
    print("="*50)


def balance_dataset(data_root, target_ratio=0.5):
    print(f"\nBalancing dataset to {target_ratio*100:.0f}% watermark ratio...")

    for split in ["train", "val", "test"]:
        wm_path = os.path.join(data_root, split, "watermark")
        no_wm_path = os.path.join(data_root, split, "no_watermark")

        wm_images = [f for f in os.listdir(wm_path) if os.path.isfile(os.path.join(wm_path, f))]
        no_wm_images = [f for f in os.listdir(no_wm_path) if os.path.isfile(os.path.join(no_wm_path, f))]

        total = len(wm_images) + len(no_wm_images)
        target_wm = int(total * target_ratio)
        target_no_wm = total - target_wm

        if len(wm_images) > target_wm:
            remove_count = len(wm_images) - target_wm
            to_remove = random.sample(wm_images, remove_count)
            for f in to_remove:
                os.remove(os.path.join(wm_path, f))
            print(f"  {split}: Removed {remove_count} watermark images")

        if len(no_wm_images) > target_no_wm:
            remove_count = len(no_wm_images) - target_no_wm
            to_remove = random.sample(no_wm_images, remove_count)
            for f in to_remove:
                os.remove(os.path.join(no_wm_path, f))
            print(f"  {split}: Removed {remove_count} no_watermark images")

    verify_dataset(data_root)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare watermark datasets")
    parser.add_argument("--data_root", type=str, default=None,
                       help="Root directory for dataset (default: ./data in script dir)")
    parser.add_argument("--dataset", type=str, default="clwd",
                       choices=["pita", "clwd", "all"],
                       help="Dataset to download")
    parser.add_argument("--clwd_zip", type=str, default=None,
                       help="Path to CLWD zip file (if already downloaded)")
    parser.add_argument("--no_watermark_source", type=str, default="clwd",
                       choices=["clwd", "coco", "both"],
                       help="Source for no_watermark images")
    parser.add_argument("--coco_dir", type=str, default=None,
                       help="Path to COCO dataset (if using coco)")
    parser.add_argument("--balance", action="store_true",
                       help="Balance dataset to 50/50")
    parser.add_argument("--num_no_wm_coco", type=int, default=5000,
                       help="Number of no_watermark images from COCO")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_root is None:
        args.data_root = os.path.join(script_dir, "data")

    os.makedirs(args.data_root, exist_ok=True)
    print(f"Data root: {os.path.abspath(args.data_root)}")

    setup_folders(args.data_root)

    if args.dataset in ["clwd", "all"]:
        print("\n" + "="*50)
        print("Processing CLWD Dataset")
        print("="*50)

        clwd_path = download_and_extract_clwd(args.data_root)
        prepare_clwd_dataset(args.data_root, clwd_path)

    if args.dataset in ["pita", "all"]:
        print("\n" + "="*50)
        print("Processing PITA Dataset")
        print("="*50)
        prepare_pita_dataset(args.data_root)

    if args.no_watermark_source in ["coco", "both"]:
        create_no_watermark_from_coco(args.data_root, args.num_no_wm_coco)

    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print("="*50)

    verify_dataset(args.data_root)

    if args.balance:
        balance_dataset(args.data_root)


if __name__ == "__main__":
    main()