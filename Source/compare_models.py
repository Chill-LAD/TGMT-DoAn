import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from config import Config
from dataset import get_dataloader


def create_model(model_type):
    if model_type == "resnet18":
        from baseline_resnet import create_baseline_resnet
        return create_baseline_resnet(num_classes=2, pretrained=True, dropout=0.5)
    elif model_type == "mobilenet":
        from baseline_mobilenet import create_baseline_mobilenet
        return create_baseline_mobilenet(num_classes=2, pretrained=True, dropout=0.5)
    elif model_type == "hybrid":
        from model import create_model
        return create_model(num_classes=2, backbone="resnet18", pretrained=True,
                          dropout=0.5, use_se_attention=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_single_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [{model.__class__.__name__}]")
    for batch_idx, batch in enumerate(pbar):
        rgb = batch["rgb"].to(device)
        labels = batch["label"].to(device)

        if model.__class__.__name__ == "HybridWatermarkModel":
            freq = batch["frequency"].to(device)
        else:
            freq = None

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(rgb, freq) if freq is not None else model(rgb)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(rgb, freq) if freq is not None else model(rgb)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss/(batch_idx+1):.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device)
            labels = batch["label"].to(device)

            if model.__class__.__name__ == "HybridWatermarkModel":
                freq = batch["frequency"].to(device)
            else:
                freq = None

            outputs = model(rgb, freq) if freq is not None else model(rgb)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def train_model(model_type, train_dir, val_dir, num_epochs=30, batch_size=32,
                lr=1e-4, checkpoint_dir=None, resume=None):
    device = torch.device(Config.device)
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*60}")

    if checkpoint_dir is None:
        checkpoint_dir = f"./checkpoints/{model_type}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = get_dataloader(train_dir, batch_size=batch_size,
                                  num_workers=Config.num_workers, mode="train")
    val_loader = get_dataloader(val_dir, batch_size=batch_size,
                                num_workers=Config.num_workers, mode="val")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(model_type).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)

    scaler = torch.amp.GradScaler('cuda') if Config.use_amp and device.type == "cuda" else None

    start_epoch = 0
    best_val_acc = 0.0

    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    results = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_time": []
    }

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_single_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        scheduler.step()

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["epoch_time"].append(epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, best_path)
            print(f"  -> Best model saved: {best_path}")

    print(f"\n{model_type.upper()} training completed! Best val acc: {best_val_acc:.2f}%")
    return results, best_val_acc


def compare_models(train_dir, val_dir, model_types=None, num_epochs=30, batch_size=32):
    if model_types is None:
        model_types = ["resnet18", "mobilenet", "hybrid"]

    device = torch.device(Config.device)
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")

    comparison_results = {}

    for model_type in model_types:
        checkpoint_dir = f"./checkpoints/{model_type}"
        results, best_acc = train_model(
            model_type=model_type,
            train_dir=train_dir,
            val_dir=val_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir
        )
        comparison_results[model_type] = {
            "best_val_acc": best_acc,
            "final_train_acc": results["train_acc"][-1] if results["train_acc"] else 0,
            "avg_epoch_time": sum(results["epoch_time"]) / len(results["epoch_time"]) if results["epoch_time"] else 0
        }

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Best Val Acc':<15} {'Final Train Acc':<18} {'Avg Time/epoch':<15}")
    print("-"*60)
    for model_type, results in comparison_results.items():
        print(f"{model_type:<15} {results['best_val_acc']:.2f}%{'':10} "
              f"{results['final_train_acc']:.2f}%{'':10} "
              f"{results['avg_epoch_time']:.1f}s")

    return comparison_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train and compare watermark detection models")
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--val_dir", type=str, default="./data/val")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet", "hybrid"],
                       help="Models to train: resnet18, mobilenet, hybrid")
    parser.add_argument("--train_single", type=str, default=None,
                       help="Train a single model: resnet18, mobilenet, or hybrid")

    args = parser.parse_args()

    if args.train_single:
        train_model(args.train_single, args.train_dir, args.val_dir,
                    num_epochs=args.epochs, batch_size=args.batch_size)
    else:
        compare_models(args.train_dir, args.val_dir, model_types=args.models,
                      num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()