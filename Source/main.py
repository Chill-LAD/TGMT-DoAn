import os
import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

from config import Config, ModelConfig
from model import create_model, WatermarkDetector
from dataset import compute_fft_spectrum
import torch.nn.functional as F
from torchvision import transforms


class WatermarkDetectionApp:
    def __init__(self, model_path=None):
        self.device = torch.device(Config.device)
        self.model = create_model(num_classes=2, backbone=ModelConfig.backbone,
                                  pretrained=False, dropout=ModelConfig.dropout,
                                  use_se_attention=True)
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
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = Config.image_size, Config.image_size
        image_resized = cv2.resize(image, (w, h))

        rgb_tensor = torch.from_numpy(image_resized).float() / 255.0
        rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(rgb_tensor.permute(2, 0, 1))
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)

        freq_tensor = compute_fft_spectrum(image).unsqueeze(0).to(self.device)

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
    from train import train
    train(args.train_dir, args.val_dir, num_epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, backbone=args.backbone,
          resume=args.resume)


def eval_command(args):
    from evaluate import evaluate, evaluate_on_conditions

    if args.conditions:
        evaluate_on_conditions(args.model_path, args.test_dir, args.conditions)
    else:
        evaluate(args.model_path, args.test_dir, args.batch_size, args.tta)


def detect_command(args):
    app = WatermarkDetectionApp(args.model)

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
    parser = argparse.ArgumentParser(
        description="Watermark Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:
    python main.py train --train_dir ./data/train --val_dir ./data/val --epochs 30

  Evaluate model:
    python main.py eval --model_path ./checkpoints/best_model.pth --test_dir ./data/test

  Detect single image:
    python main.py detect --model ./checkpoints/best_model.pth --input ./test/image.jpg

  Batch detection:
    python main.py detect --model ./checkpoints/best_model.pth --input ./test_folder --output results.txt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train watermark detection model")
    train_parser.add_argument("--train_dir", type=str, default="./data/train")
    train_parser.add_argument("--val_dir", type=str, default="./data/val")
    train_parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    train_parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    train_parser.add_argument("--lr", type=float, default=Config.learning_rate)
    train_parser.add_argument("--backbone", type=str, default="resnet18",
                             choices=["resnet18", "resnet34", "resnet50"])
    train_parser.add_argument("--resume", type=str, default=None)

    eval_parser = subparsers.add_parser("eval", help="Evaluate watermark detection model")
    eval_parser.add_argument("--model_path", type=str, required=True)
    eval_parser.add_argument("--test_dir", type=str, default="./data/test")
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    eval_parser.add_argument("--conditions", nargs="+",
                             default=["clean", "jpeg", "resize", "noise"])

    detect_parser = subparsers.add_parser("detect", help="Detect watermark in image(s)")
    detect_parser.add_argument("--model", type=str, required=True)
    detect_parser.add_argument("--input", type=str, required=True,
                              help="Image file or directory")
    detect_parser.add_argument("--output", type=str, default=None,
                              help="Output file for batch results")
    detect_parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "detect":
        detect_command(args)


if __name__ == "__main__":
    main()