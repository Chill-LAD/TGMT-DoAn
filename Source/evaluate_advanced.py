"""
Optimized script đánh giá tác động của TTA và Multi-scale.
Sử dụng batched operations thay vì per-image để tăng tốc.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config import Config, ModelConfig
from dual_dataset import get_dual_dataloader
from detect import WatermarkDetector
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def evaluate_batched(detector, dataloader, use_tta=False, multi_scale=False, device="cuda", desc=""):
    """
    Đánh giá model với TTA/Multi-scale sử dụng batched operations.
    """
    all_preds = []
    all_labels = []
    inference_times = []

    for batch in tqdm(dataloader, desc=desc):
        rgb_batch = batch["rgb"].to(device)
        freq_batch = batch["frequency"].to(device)
        labels = batch["label"].to(device)

        batch_start = time.time()
        probs_list = []

        # Multi-scale: chạy ở 224 và 448
        scales = [448, 224] if multi_scale else [224]

        for scale in scales:
            if scale == 224:
                rgb_resized = rgb_batch
            else:
                rgb_resized = F.interpolate(rgb_batch, size=(scale, scale), mode='bilinear', align_corners=False)

            # Original
            with torch.no_grad():
                logits = detector.model(rgb_resized, freq_batch)
                probs_list.append(F.softmax(logits, dim=1))

            if use_tta:
                # Flip
                rgb_flip = torch.flip(rgb_resized, dims=[3])
                with torch.no_grad():
                    logits = detector.model(rgb_flip, freq_batch)
                    probs_list.append(F.softmax(logits, dim=1))

                # Rotation +10
                rgb_rot = TF.rotate(rgb_resized, 10, fill=0)
                with torch.no_grad():
                    logits = detector.model(rgb_rot, freq_batch)
                    probs_list.append(F.softmax(logits, dim=1))

                # Rotation -10
                rgb_rot = TF.rotate(rgb_resized, -10, fill=0)
                with torch.no_grad():
                    logits = detector.model(rgb_rot, freq_batch)
                    probs_list.append(F.softmax(logits, dim=1))

        probs = torch.stack(probs_list).mean(dim=0)
        batch_time = time.time() - batch_start
        inference_times.append(batch_time / rgb_batch.size(0))

        _, predicted = probs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    avg_time = np.mean(inference_times) * 1000  # ms per image

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_ms": avg_time,
    }


def main():
    """Chạy ablation study cho TTA và Multi-scale."""
    device = torch.device(Config.device)
    print(f"Device: {device}\n")

    # Load test dataloader
    print("Loading test dataloader...")
    test_loader = get_dual_dataloader(
        visible_dir="./data/test",
        invisible_dir="./data_invisible/test",
        batch_size=32,
        num_workers=4,
        mode="test",
        merge=True
    )
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Initialize detector
    model_path = "./checkpoints/ours_v2_combined/best_model.pth"
    detector = WatermarkDetector(model_path, model_version="v2")
    detector.model.to(device)

    results = {}

    # 1. Baseline
    print("="*60)
    print("[1/4] Baseline (no TTA, no Multi-scale)")
    print("="*60)
    results["baseline"] = evaluate_batched(
        detector, test_loader, use_tta=False, multi_scale=False, device=device, desc="Baseline"
    )
    print(f"Result: Acc={results['baseline']['accuracy']*100:.2f}%, F1={results['baseline']['f1']*100:.2f}%, Time={results['baseline']['inference_ms']:.1f}ms\n")

    # 2. TTA only
    print("="*60)
    print("[2/4] TTA only (flip + rotation)")
    print("="*60)
    results["tta"] = evaluate_batched(
        detector, test_loader, use_tta=True, multi_scale=False, device=device, desc="TTA"
    )
    print(f"Result: Acc={results['tta']['accuracy']*100:.2f}%, F1={results['tta']['f1']*100:.2f}%, Time={results['tta']['inference_ms']:.1f}ms\n")

    # 3. Multi-scale only
    print("="*60)
    print("[3/4] Multi-scale only (224 + 448)")
    print("="*60)
    results["multi_scale"] = evaluate_batched(
        detector, test_loader, use_tta=False, multi_scale=True, device=device, desc="Multi-scale"
    )
    print(f"Result: Acc={results['multi_scale']['accuracy']*100:.2f}%, F1={results['multi_scale']['f1']*100:.2f}%, Time={results['multi_scale']['inference_ms']:.1f}ms\n")

    # 4. TTA + Multi-scale
    print("="*60)
    print("[4/4] TTA + Multi-scale")
    print("="*60)
    results["tta_ms"] = evaluate_batched(
        detector, test_loader, use_tta=True, multi_scale=True, device=device, desc="TTA+MS"
    )
    print(f"Result: Acc={results['tta_ms']['accuracy']*100:.2f}%, F1={results['tta_ms']['f1']*100:.2f}%, Time={results['tta_ms']['inference_ms']:.1f}ms\n")

    # Print comparison
    print("="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Method':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Inference':<12}")
    print("-"*72)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']*100:.2f}%{'':<3} {r['f1']*100:.2f}%{'':<3} "
              f"{r['precision']*100:.2f}%{'':<3} {r['recall']*100:.2f}%{'':<3} {r['inference_ms']:.1f}ms")

    # Print deltas
    print("\n" + "="*60)
    print("IMPROVEMENT vs BASELINE")
    print("="*60)
    base = results["baseline"]
    for name in ["tta", "multi_scale", "tta_ms"]:
        r = results[name]
        print(f"{name}:")
        print(f"  Accuracy: {base['accuracy']*100:.2f}% -> {r['accuracy']*100:.2f}% ({(r['accuracy']-base['accuracy'])*100:+.2f}%)")
        print(f"  F1:       {base['f1']*100:.2f}% -> {r['f1']*100:.2f}% ({(r['f1']-base['f1'])*100:+.2f}%)")
        print(f"  Precision: {base['precision']*100:.2f}% -> {r['precision']*100:.2f}% ({(r['precision']-base['precision'])*100:+.2f}%)")
        print(f"  Recall:   {base['recall']*100:.2f}% -> {r['recall']*100:.2f}% ({(r['recall']-base['recall'])*100:+.2f}%)")
        print(f"  Inference: {base['inference_ms']:.1f}ms -> {r['inference_ms']:.1f}ms (x{r['inference_ms']/base['inference_ms']:.1f})")
        print()

    # Save results
    with open("ablation_results.txt", "w") as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"{'Method':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Inference':<12}\n")
        f.write("-"*72 + "\n")
        for name, r in results.items():
            f.write(f"{name:<20} {r['accuracy']*100:.2f}%{'':<3} {r['f1']*100:.2f}%{'':<3} "
                    f"{r['precision']*100:.2f}%{'':<3} {r['recall']*100:.2f}%{'':<3} {r['inference_ms']:.1f}ms\n")
    print("Results saved to ablation_results.txt")


if __name__ == "__main__":
    main()
