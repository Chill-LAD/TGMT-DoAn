"""
Script đánh giá mô hình hybrid watermark detection.
Tính các metrics: accuracy, precision, recall, F1-score.
Hỗ trợ Test-Time Augmentation (TTA).
"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import torch.nn.functional as F

from config import Config, ModelConfig


def create_model(num_classes=2, backbone="resnet18", pretrained=False, dropout=0.5, use_se_attention=True):
    """Tạo hybrid model để đánh giá."""
    from model import create_model
    return create_model(num_classes=num_classes, backbone=backbone,
                       pretrained=pretrained, dropout=dropout, use_se_attention=use_se_attention)


def get_test_dataloader(visible_test_dir=None, invisible_test_dir=None,
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


def evaluate(model_path, visible_test_dir=None, invisible_test_dir=None,
           batch_size=32, use_tta=False, merge=True):
    """
    Đánh giá model trên test set.
    Args:
        model_path: Đường dẫn đến checkpoint
        visible_test_dir: Đường dẫn visible watermark test data
        invisible_test_dir: Đường dẫn invisible watermark test data
        batch_size: Kích thước batch
        use_tta: Có sử dụng Test-Time Augmentation không
        merge: Có merge cả 2 datasets không
    Returns:
        Dict chứa các metrics
    """
    device = torch.device(Config.device)
    print(f"Using device: {device}")
    print(f"Visible test: {visible_test_dir}")
    print(f"Invisible test: {invisible_test_dir}")
    print(f"Merge datasets: {merge}")

    # Tạo và load model
    model = create_model(num_classes=2, backbone=ModelConfig.backbone,
                         pretrained=False, dropout=ModelConfig.dropout,
                         use_se_attention=True)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded model from {model_path}")

    # Tạo test dataloader
    test_loader = get_test_dataloader(
        visible_test_dir=visible_test_dir,
        invisible_test_dir=invisible_test_dir,
        batch_size=batch_size,
        num_workers=Config.num_workers,
        merge=merge
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Running evaluation...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            rgb = batch["rgb"].to(device)
            freq = batch["frequency"].to(device)
            labels = batch["label"].to(device)

            if use_tta:
                # Test-Time Augmentation: flip horizontal
                outputs_list = []

                outputs = model(rgb, freq)
                outputs_list.append(F.softmax(outputs, dim=1))

                rgb_flip = torch.flip(rgb, dims=[3])
                outputs = model(rgb_flip, freq)
                outputs_list.append(F.softmax(outputs, dim=1))

                probs = torch.stack(outputs_list).mean(dim=0)
            else:
                outputs = model(rgb, freq)
                probs = F.softmax(outputs, dim=1)

            _, predicted = probs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
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

    # In kết quả
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*50)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                               target_names=['No Watermark', 'Watermark']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def evaluate_on_classes(model_path, test_dir, classes=None):
    """
    Đánh giá model trên từng class riêng biệt.
    Args:
        model_path: Đường dẫn đến checkpoint
        test_dir: Đường dẫn test data
        classes: Danh sách các classes cần đánh giá
    """
    if classes is None:
        classes = ["watermark", "no_watermark"]

    results = {}

    for class_name in classes:
        print(f"\nEvaluating on class: {class_name}")
        class_dir = os.path.join(test_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"  Class dir not found: {class_dir}")
            continue

        result = evaluate(model_path, visible_test_dir=test_dir,
                        invisible_test_dir=None, merge=True)
        results[class_name] = result

    print("\n" + "="*50)
    print("SUMMARY BY CLASS")
    print("="*50)
    for class_name, result in results.items():
        print(f"{class_name:15s} - F1: {result['f1']*100:.2f}%, "
              f"Acc: {result['accuracy']*100:.2f}%")

    return results


def main():
    """Entry point cho command line interface."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Watermark Detection Model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--visible_test_dir", type=str, default="./data/test",
                       help="Path to visible watermark test data")
    parser.add_argument("--invisible_test_dir", type=str, default="./data_invisible/test",
                       help="Path to invisible watermark test data")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--tta", action="store_true",
                       help="Use test-time augmentation")
    parser.add_argument("--no_merge", action="store_true",
                       help="Don't merge visible and invisible datasets")
    parser.add_argument("--classes", nargs="+", default=None,
                       help="Classes to evaluate (e.g., watermark no_watermark)")

    args = parser.parse_args()

    merge = not args.no_merge

    if args.classes:
        evaluate_on_classes(args.model_path, args.visible_test_dir, args.classes)
    else:
        evaluate(args.model_path,
                visible_test_dir=args.visible_test_dir,
                invisible_test_dir=args.invisible_test_dir,
                batch_size=args.batch_size,
                use_tta=args.tta,
                merge=merge)


if __name__ == "__main__":
    main()