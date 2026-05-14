"""
CLI interface cho bài toán phát hiện watermark.
Cung cấp 3 lệnh chính: train, eval, detect.
Hỗ trợ cả visible và invisible watermarks.
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*pretrained.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

from config import Config, ModelConfig
import torch.nn.functional as F
from torchvision import transforms


class WatermarkDetectionApp:
    """
    Ứng dụng phát hiện watermark với giao diện CLI.
    Hỗ trợ phát hiện ảnh đơn lẻ và hàng loạt.
    """
    def __init__(self, model_path=None, model_version="v2"):
        from model_v1 import create_model_v1
        from model_v2 import create_model_v2
        from dataset import compute_fft_spectrum

        self.compute_fft_spectrum = compute_fft_spectrum
        self.device = torch.device(Config.device)

        if model_version == "v1":
            self.model = create_model_v1(
                num_classes=2,
                backbone=ModelConfig.backbone,
                pretrained=False,
                dropout=ModelConfig.dropout
            )
        else:
            self.model = create_model_v2(
                num_classes=2,
                backbone=ModelConfig.backbone,
                pretrained=False,
                dropout=ModelConfig.dropout
            )

        self.model.to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model loaded. Using untrained model.")

    def detect_single_image(self, image_path, use_tta=False):
        """
        Phát hiện watermark trong một ảnh.
        Args:
            image_path: Đường dẫn đến ảnh
            use_tta: Có sử dụng Test-Time Augmentation không
        Returns:
            Dict chứa label, confidence, và probabilities
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize ảnh
        h, w = Config.image_size, Config.image_size
        image_resized = cv2.resize(image, (w, h))

        # Prepare RGB tensor
        rgb_tensor = torch.from_numpy(image_resized).float() / 255.0
        rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(rgb_tensor.permute(2, 0, 1))
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)

        # Prepare frequency tensor
        freq_tensor = self.compute_fft_spectrum(image).unsqueeze(0).to(self.device)

        # Inference với TTA
        if use_tta:
            probs_list = []

            with torch.no_grad():
                logits = self.model(rgb_tensor, freq_tensor)
                probs_list.append(F.softmax(logits, dim=1))

                rgb_flip = torch.flip(rgb_tensor, dims=[3])
                logits = self.model(rgb_flip, freq_tensor)
                probs_list.append(F.softmax(logits, dim=1))

            probs = torch.stack(probs_list).mean(dim=0)
        else:
            with torch.no_grad():
                logits = self.model(rgb_tensor, freq_tensor)
                probs = F.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

        label = "Watermark" if pred == 1 else "No Watermark"

        return {
            "label": label,
            "confidence": confidence,
            "prob_no_wm": probs[0, 0].item(),
            "prob_wm": probs[0, 1].item()
        }

    def detect_batch(self, image_dir, output_file=None):
        """
        Phát hiện watermark trong nhiều ảnh.
        Args:
            image_dir: Thư mục chứa ảnh
            output_file: Đường dẫn file lưu kết quả
        Returns:
            List chứa kết quả của từng ảnh
        """
        image_files = [f for f in os.listdir(image_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        results = []
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                result = self.detect_single_image(img_path)
                result["filename"] = img_file
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        if output_file:
            with open(output_file, 'w') as f:
                for r in results:
                    f.write(f"{r['filename']}: {r['label']} ({r['confidence']:.4f})\n")

        return results


def train_command(args):
    """Xử lý lệnh train."""
    from train_v1 import train
    train(visible_train_dir=args.visible_train_dir,
          visible_val_dir=args.visible_val_dir,
          invisible_train_dir=args.invisible_train_dir,
          invisible_val_dir=args.invisible_val_dir,
          num_epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          backbone=args.backbone,
          checkpoint_dir=args.checkpoint_dir,
          resume=args.resume,
          merge=not args.no_merge)


def eval_command(args):
    """Xử lý lệnh eval."""
    from evaluate import evaluate, evaluate_on_classes

    if args.classes is not None:
        evaluate_on_classes(args.model_path, args.visible_test_dir, args.classes)
    else:
        evaluate(args.model_path,
                visible_test_dir=args.visible_test_dir,
                invisible_test_dir=args.invisible_test_dir,
                batch_size=args.batch_size,
                use_tta=args.tta,
                merge=not args.no_merge)


def detect_command(args):
    """Xử lý lệnh detect."""
    app = WatermarkDetectionApp(args.model, args.model_version)

    if os.path.isfile(args.input):
        result = app.detect_single_image(args.input, args.tta)
        print(f"Image: {args.input}")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  P(No Watermark): {result['prob_no_wm']*100:.2f}%")
        print(f"  P(Watermark): {result['prob_wm']*100:.2f}%")

    elif os.path.isdir(args.input):
        results = app.detect_batch(args.input, args.output)
        print(f"Processed {len(results)} images")
        if args.output:
            print(f"Results saved to {args.output}")


def main():
    """
    Main entry point cho CLI.
    Cung cấp 3 subcommands: train, eval, detect.
    """
    parser = argparse.ArgumentParser(
        description="Watermark Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model (combined dataset):
    python main.py train --epochs 30

  Train model (visible only):
    python main.py train --visible_train_dir ./data/train --visible_val_dir ./data/val --no_merge

  Evaluate model:
    python main.py eval --model_path ./checkpoints/best_model.pth

  Detect single image:
    python main.py detect --model ./checkpoints/best_model.pth --input ./test/image.jpg
        """
    )

    # Subparsers cho các lệnh
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train watermark detection model")
    train_parser.add_argument("--visible_train_dir", type=str, default="./data/train",
                            help="Path to visible watermark training data")
    train_parser.add_argument("--visible_val_dir", type=str, default="./data/val",
                            help="Path to visible watermark validation data")
    train_parser.add_argument("--invisible_train_dir", type=str, default="./data_invisible/train",
                            help="Path to invisible watermark training data")
    train_parser.add_argument("--invisible_val_dir", type=str, default="./data_invisible/val",
                            help="Path to invisible watermark validation data")
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ours_v1_combined",
                            help="Path to save checkpoints")
    train_parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    train_parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    train_parser.add_argument("--lr", type=float, default=Config.learning_rate)
    train_parser.add_argument("--backbone", type=str, default="resnet18",
                             choices=["resnet18", "resnet34", "resnet50"])
    train_parser.add_argument("--resume", type=str, default=None)
    train_parser.add_argument("--no_merge", action="store_true",
                             help="Don't merge visible and invisible datasets")

    # Eval parser
    eval_parser = subparsers.add_parser("eval", help="Evaluate watermark detection model")
    eval_parser.add_argument("--model_path", type=str, required=True)
    eval_parser.add_argument("--visible_test_dir", type=str, default="./data/test",
                           help="Path to visible watermark test data")
    eval_parser.add_argument("--invisible_test_dir", type=str, default="./data_invisible/test",
                           help="Path to invisible watermark test data")
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    eval_parser.add_argument("--no_merge", action="store_true",
                            help="Don't merge visible and invisible datasets")
    eval_parser.add_argument("--classes", nargs="+", default=None,
                            help="Classes to evaluate (e.g., watermark no_watermark)")

    # Detect parser
    detect_parser = subparsers.add_parser("detect", help="Detect watermark in image(s)")
    detect_parser.add_argument("--model", type=str, required=True)
    detect_parser.add_argument("--input", type=str, required=True,
                              help="Image file or directory")
    detect_parser.add_argument("--output", type=str, default=None,
                              help="Output file for batch results")
    detect_parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    detect_parser.add_argument("--model_version", type=str, default="v2",
                              choices=["v1", "v2"],
                              help="Model version (v1: no SE, v2: with SE)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Dispatch commands
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "detect":
        detect_command(args)


if __name__ == "__main__":
    main()