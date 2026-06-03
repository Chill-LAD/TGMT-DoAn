"""
CLI interface cho phát hiện watermark.
Hỗ trợ phát hiện ảnh đơn lẻ và hàng loạt với TTA.
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*pretrained.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from config import Config, ModelConfig
import torch.nn.functional as F
from torchvision import transforms


class WatermarkDetector:
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


def detect_command(args):
    """Xử lý lệnh detect."""
    app = WatermarkDetector(args.model, args.model_version)

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
    Main entry point cho CLI detect.
    """
    parser = argparse.ArgumentParser(
        description="Watermark Detection CLI - Detect mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Detect single image (v2 with SE Attention):
    python detect.py --model ./checkpoints/ours_v2_combined/best_model.pth --input ./test.jpg --model_version v2

  Detect single image (v1 without SE Attention):
    python detect.py --model ./checkpoints/ours_v1_combined/best_model.pth --input ./test.jpg --model_version v1

  Detect batch images:
    python detect.py --model ./checkpoints/ours_v2_combined/best_model.pth --input ./test_folder --output results.txt --model_version v2

  Detect with TTA (Test-Time Augmentation):
    python detect.py --model ./checkpoints/ours_v2_combined/best_model.pth --input ./test.jpg --tta --model_version v2
        """
    )

    # Detect parser
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Image file or directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for batch results")
    parser.add_argument("--tta", action="store_true",
                       help="Use test-time augmentation")
    parser.add_argument("--model_version", type=str, default="v2",
                       choices=["v1", "v2"],
                       help="Model version (v1: no SE, v2: with SE)")

    args = parser.parse_args()
    detect_command(args)


if __name__ == "__main__":
    main()
