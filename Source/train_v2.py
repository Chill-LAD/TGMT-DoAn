"""
Script huấn luyện mô hình Ours v2 (Hybrid CNN-Frequency với SE Attention).
Huấn luyện trên visible, invisible hoặc combined dataset.
"""
import os
import time
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config


def set_seed(seed):
    """Đặt seed cho reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def get_dataloaders(visible_train_dir=None, visible_val_dir=None,
                   invisible_train_dir=None, invisible_val_dir=None,
                   batch_size=32, num_workers=4, merge=True):
    """Tạo train và validation dataloaders."""
    from dual_dataset import get_dual_dataloader

    train_loader = get_dual_dataloader(
        visible_dir=visible_train_dir,
        invisible_dir=invisible_train_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="train",
        merge=merge
    )

    val_loader = get_dual_dataloader(
        visible_dir=visible_val_dir,
        invisible_dir=invisible_val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="val",
        merge=merge
    )

    return train_loader, val_loader


def create_model(num_classes=2, backbone="resnet18", pretrained=True, dropout=0.5):
    """Tạo Ours v2 model (có SE attention)."""
    from model_v2 import create_model_v2
    return create_model_v2(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    )


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Huấn luyện một epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        rgb = batch["rgb"].to(device)
        freq = batch["frequency"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(rgb, freq)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(rgb, freq)
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
    """Đánh giá model trên validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Validation]"):
            rgb = batch["rgb"].to(device)
            freq = batch["frequency"].to(device)
            labels = batch["label"].to(device)

            outputs = model(rgb, freq)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def train(visible_train_dir=None, visible_val_dir=None,
         invisible_train_dir=None, invisible_val_dir=None,
         num_epochs=30, batch_size=32, lr=1e-4,
         backbone="resnet18", checkpoint_dir="./checkpoints/ours_v2_combined", resume=None, merge=True):
    """
    Hàm huấn luyện chính cho Ours v2.
    """
    set_seed(Config.seed)

    device = torch.device(Config.device)
    print(f"Using device: {device}")
    print(f"Model: Ours v2 (Hybrid CNN-Frequency + SE Attention)")
    print(f"Visible train: {visible_train_dir}")
    print(f"Invisible train: {invisible_train_dir}")
    print(f"Merge datasets: {merge}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        visible_train_dir=visible_train_dir,
        visible_val_dir=visible_val_dir,
        invisible_train_dir=invisible_train_dir,
        invisible_val_dir=invisible_val_dir,
        batch_size=batch_size,
        num_workers=Config.num_workers,
        merge=merge
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(
        num_classes=2,
        backbone=backbone,
        pretrained=True,
        dropout=0.5
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)

    scaler = torch.amp.GradScaler('cuda') if Config.use_amp and device.type == "cuda" else None

    start_epoch = 0
    best_val_acc = 0.0

    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(checkpoint_dir + "_logs")

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
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
            print(f"  Saved best model: {best_path}")

    writer.close()
    print(f"Training completed! Best val acc: {best_val_acc:.2f}%")
    return best_val_acc


def main():
    """Entry point cho command line interface."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Ours v2 Watermark Detection Model")
    parser.add_argument("--visible_train_dir", type=str, default="./data/train",
                       help="Path to visible watermark training data")
    parser.add_argument("--visible_val_dir", type=str, default="./data/val",
                       help="Path to visible watermark validation data")
    parser.add_argument("--invisible_train_dir", type=str, default="./data_invisible/train",
                       help="Path to invisible watermark training data")
    parser.add_argument("--invisible_val_dir", type=str, default="./data_invisible/val",
                       help="Path to invisible watermark validation data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ours_v2_combined",
                       help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.learning_rate,
                       help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50"],
                       help="Backbone architecture")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--no_merge", action="store_true",
                       help="Don't merge visible and invisible datasets")

    args = parser.parse_args()

    merge = not args.no_merge

    if args.resume:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"Auto-detected checkpoint_dir from resume: {checkpoint_dir}")
    else:
        checkpoint_dir = args.checkpoint_dir

    train(
        visible_train_dir=args.visible_train_dir,
        visible_val_dir=args.visible_val_dir,
        invisible_train_dir=args.invisible_train_dir,
        invisible_val_dir=args.invisible_val_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        checkpoint_dir=checkpoint_dir,
        resume=args.resume,
        merge=merge
    )


if __name__ == "__main__":
    main()