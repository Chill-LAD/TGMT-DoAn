import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import time

from config import Config, ModelConfig


def create_model(model_type, pretrained=False):
    if model_type == "resnet18":
        from baseline_resnet import create_baseline_resnet
        return create_baseline_resnet(num_classes=2, pretrained=pretrained, dropout=0.5)
    elif model_type == "mobilenet":
        from baseline_mobilenet import create_baseline_mobilenet
        return create_baseline_mobilenet(num_classes=2, pretrained=pretrained, dropout=0.5)
    elif model_type == "hybrid":
        from model import create_model
        return create_model(num_classes=2, backbone="resnet18", pretrained=pretrained,
                          dropout=0.5, use_se_attention=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_single_model(model_type, model_path, test_dir, batch_size=32, use_tta=False):
    device = torch.device(Config.device)
    print(f"\nEvaluating {model_type.upper()}...")

    model = create_model(model_type, pretrained=False).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    from dataset import get_dataloader
    test_loader = get_dataloader(test_dir, batch_size=batch_size,
                                num_workers=Config.num_workers, mode="test")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []

    print(f"Test samples: {len(test_loader.dataset)}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{model_type}]"):
            rgb = batch["rgb"].to(device)

            if model_type == "hybrid":
                freq = batch["frequency"].to(device)
            else:
                freq = None

            start_time = time.time()
            if freq is not None:
                outputs = model(rgb, freq)
            else:
                outputs = model(rgb)
            inference_times.append(time.time() - start_time)

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
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)
    avg_inference = np.mean(inference_times) * 1000

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


def evaluate_all_models(checkpoint_dir, test_dir, batch_size=32, use_tta=False):
    results = {}

    for model_type in ["resnet18", "mobilenet", "hybrid"]:
        model_path = os.path.join(checkpoint_dir, model_type, "best_model.pth")

        if os.path.exists(model_path):
            result = evaluate_single_model(model_type, model_path, test_dir, batch_size, use_tta)
            results[model_type] = result
        else:
            print(f"Model not found: {model_path}")

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
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate watermark detection models")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory containing model checkpoints")
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--model", type=str, default=None,
                       choices=["resnet18", "mobilenet", "hybrid", "all"],
                       help="Specific model to evaluate (default: all)")

    args = parser.parse_args()

    if args.model and args.model != "all":
        model_path = os.path.join(args.checkpoint_dir, args.model, "best_model.pth")
        if os.path.exists(model_path):
            evaluate_single_model(args.model, model_path, args.test_dir,
                                args.batch_size, args.tta)
        else:
            print(f"Model not found: {model_path}")
    else:
        evaluate_all_models(args.checkpoint_dir, args.test_dir, args.batch_size, args.tta)


if __name__ == "__main__":
    main()