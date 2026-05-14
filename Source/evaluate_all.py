"""
Script đánh giá tất cả các mô hình watermark detection và so sánh kết quả.
Hỗ trợ ResNet18, MobileNetV3, Ours v1 và Ours v2.
Hỗ trợ Test-Time Augmentation (TTA).
"""
import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import time

from config import Config, ModelConfig


def create_model(model_type, pretrained=False):
    """
    Tạo model theo loại.
    Args:
        model_type: resnet18, mobilenet, ours_v1, ours_v2
        pretrained: Có load pretrained weights không
    """
    if model_type == "resnet18":
        from baseline_resnet import create_baseline_resnet
        return create_baseline_resnet(num_classes=2, pretrained=pretrained, dropout=0.5)
    elif model_type == "mobilenet":
        from baseline_mobilenet import create_baseline_mobilenet
        return create_baseline_mobilenet(num_classes=2, pretrained=pretrained, dropout=0.5)
    elif model_type == "ours_v1":
        from model_v1 import create_model_v1
        return create_model_v1(num_classes=2, backbone="resnet18", pretrained=pretrained, dropout=0.5)
    elif model_type == "ours_v2":
        from model_v2 import create_model_v2
        return create_model_v2(num_classes=2, backbone="resnet18", pretrained=pretrained, dropout=0.5)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: resnet18, mobilenet, ours_v1, ours_v2")


def get_model_needs_freq(model_type):
    """Kiểm tra model có cần frequency input không."""
    return model_type in ["ours_v1", "ours_v2"]


def get_test_dataloaders(visible_test_dir=None, invisible_test_dir=None,
                         batch_size=32, num_workers=4, merge=True):
    """Tạo test dataloader."""
    from dual_dataset import get_dual_dataloader

    test_loader = get_dual_dataloader(
        visible_dir=visible_test_dir,
        invisible_dir=invisible_test_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="test",
        merge=merge
    )

    return test_loader


def evaluate_single_model(model_type, model_path, visible_test_dir=None,
                          invisible_test_dir=None, batch_size=32, use_tta=False, merge=True):
    """
    Đánh giá một model cụ thể.
    Trả về các metrics và thời gian inference.
    """
    device = torch.device(Config.device)
    print(f"\nEvaluating {model_type.upper()}...")

    model = create_model(model_type, pretrained=False).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    test_loader = get_test_dataloaders(
        visible_test_dir=visible_test_dir,
        invisible_test_dir=invisible_test_dir,
        batch_size=batch_size,
        num_workers=Config.num_workers,
        merge=merge
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_types = []
    inference_times = []

    print(f"Test samples: {len(test_loader.dataset)}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{model_type}]"):
            rgb = batch["rgb"].to(device)

            if get_model_needs_freq(model_type):
                freq = batch["frequency"].to(device)
            else:
                freq = None

            start_time = time.time()
            if freq is not None:
                outputs = model(rgb, freq)
            else:
                outputs = model(rgb)
            inference_times.append(time.time() - start_time)

            # Test-Time Augmentation
            if use_tta:
                outputs_list = [outputs]

                if freq is not None:
                    rgb_flip = torch.flip(rgb, dims=[3])
                    outputs_list.append(model(rgb_flip, freq))
                else:
                    rgb_flip = torch.flip(rgb, dims=[3])
                    outputs_list.append(model(rgb_flip))

                probs = torch.stack(outputs_list).mean(dim=0)
                probs = torch.softmax(probs, dim=1)
            else:
                probs = torch.softmax(outputs, dim=1)

            _, predicted = probs.max(1)

            labels = batch["label"].numpy()
            types = batch.get("type", ["unknown"] * len(labels))
            if isinstance(types, list):
                all_types.extend(types)
            else:
                all_types.extend(types.tolist() if hasattr(types, 'tolist') else [types] * len(labels))

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Tính metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)
    avg_inference = np.mean(inference_times) * 1000

    # In kết quả
    print(f"\n{model_type.upper()} Results:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  Inference:  {avg_inference:.1f}ms/image")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                               target_names=['No Watermark', 'Watermark']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'inference_ms': avg_inference,
        'confusion_matrix': cm
    }


def evaluate_all_models(checkpoint_dir, visible_test_dir=None, invisible_test_dir=None,
                        batch_size=32, use_tta=False, merge=True):
    """
    Đánh giá tất cả các mô hình đã huấn luyện.
    In bảng so sánh kết quả cuối cùng.
    """
    results = {}

    for model_type in ["resnet18", "mobilenet", "ours_v1", "ours_v2"]:
        model_path = os.path.join(checkpoint_dir, f"{model_type}_combined", "best_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_dir, model_type, "best_model.pth")

        if os.path.exists(model_path):
            result = evaluate_single_model(model_type, model_path,
                                          visible_test_dir=visible_test_dir,
                                          invisible_test_dir=invisible_test_dir,
                                          batch_size=batch_size,
                                          use_tta=use_tta,
                                          merge=merge)
            results[model_type] = result
        else:
            print(f"Model not found: {model_path}")

    # In bảng so sánh
    print(f"\n{'='*70}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Inference':<12}")
    print("-"*70)

    for model_type, result in results.items():
        print(f"{model_type:<12} {result['accuracy']*100:.2f}%{'':5} "
              f"{result['precision']*100:.2f}%{'':5} "
              f"{result['recall']*100:.2f}%{'':5} "
              f"{result['f1']*100:.2f}%{'':5} "
              f"{result['inference_ms']:.1f}ms")

    return results


def main():
    """Entry point cho command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate watermark detection models")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory containing model checkpoints")
    parser.add_argument("--visible_test_dir", type=str, default="./data/test",
                       help="Path to visible watermark test data")
    parser.add_argument("--invisible_test_dir", type=str, default="./data_invisible/test",
                       help="Path to invisible watermark test data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--no_merge", action="store_true", help="Don't merge visible and invisible datasets")
    parser.add_argument("--model", type=str, default=None,
                       choices=["resnet18", "mobilenet", "ours_v1", "ours_v2", "all"],
                       help="Specific model to evaluate (default: all)")

    args = parser.parse_args()

    merge = not args.no_merge

    if args.model and args.model != "all":
        model_path = os.path.join(args.checkpoint_dir, f"{args.model}_combined", "best_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(args.checkpoint_dir, args.model, "best_model.pth")
        if os.path.exists(model_path):
            evaluate_single_model(args.model, model_path,
                                 visible_test_dir=args.visible_test_dir,
                                 invisible_test_dir=args.invisible_test_dir,
                                 batch_size=args.batch_size,
                                 use_tta=args.tta,
                                 merge=merge)
        else:
            print(f"Model not found: {model_path}")
    else:
        evaluate_all_models(args.checkpoint_dir,
                           visible_test_dir=args.visible_test_dir,
                           invisible_test_dir=args.invisible_test_dir,
                           batch_size=args.batch_size,
                           use_tta=args.tta,
                           merge=merge)


if __name__ == "__main__":
    main()