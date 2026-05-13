import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from config import Config, ModelConfig
from dataset import get_dataloader
from model import create_model
import torch.nn.functional as F


def evaluate(model_path, test_dir, batch_size=32, use_tta=False):
    device = torch.device(Config.device)
    print(f"Using device: {device}")

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

    test_loader = get_dataloader(test_dir, batch_size=batch_size,
                                num_workers=Config.num_workers, mode="test")
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

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)

    cm = confusion_matrix(all_labels, all_preds)

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


def evaluate_on_conditions(model_path, test_dir, conditions=["clean", "jpeg", "resize", "noise"]):
    results = {}

    for condition in conditions:
        print(f"\nEvaluating on condition: {condition}")
        condition_dir = os.path.join(test_dir, condition)

        if not os.path.exists(condition_dir):
            print(f"  Condition dir not found: {condition_dir}")
            continue

        result = evaluate(model_path, condition_dir, use_tta=True)
        results[condition] = result

    print("\n" + "="*50)
    print("SUMMARY ACROSS CONDITIONS")
    print("="*50)
    for condition, result in results.items():
        print(f"{condition:15s} - F1: {result['f1']*100:.2f}%, "
              f"Acc: {result['accuracy']*100:.2f}%")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Watermark Detection Model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, default="./data/test",
                       help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--tta", action="store_true",
                       help="Use test-time augmentation")
    parser.add_argument("--conditions", nargs="+", default=None,
                       help="Test conditions to evaluate (e.g., watermark no_watermark)")

    args = parser.parse_args()

    if args.conditions:
        evaluate_on_conditions(args.model_path, args.test_dir, args.conditions)
    else:
        evaluate(args.model_path, args.test_dir, args.batch_size, args.tta)


if __name__ == "__main__":
    main()