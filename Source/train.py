import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import get_dataloader
from model import create_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
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
            with torch.cuda.amp.autocast():
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


def train(train_dir, val_dir, num_epochs=30, batch_size=32, lr=1e-4,
          backbone="resnet18", checkpoint_dir="./checkpoints", resume=None):
    set_seed(Config.seed)

    device = torch.device(Config.device)
    print(f"Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = get_dataloader(train_dir, batch_size=batch_size,
                                  num_workers=Config.num_workers, mode="train")
    val_loader = get_dataloader(val_dir, batch_size=batch_size,
                               num_workers=Config.num_workers, mode="val")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(num_classes=2, backbone=backbone,
                        pretrained=True, dropout=0.5, use_se_attention=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                      eta_min=lr/10)

    scaler = torch.cuda.amp.GradScaler() if Config.use_amp and device.type == "cuda" else None

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

    writer = SummaryWriter(Config.log_dir)

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, epoch, scaler)
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
    import argparse
    parser = argparse.ArgumentParser(description="Train Watermark Detection Model")
    parser.add_argument("--train_dir", type=str, default="./data/train",
                       help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="./data/val",
                       help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.learning_rate,
                       help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50", "mobilenet_v3_small"],
                       help="Backbone architecture")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    train(args.train_dir, args.val_dir, num_epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, backbone=args.backbone,
          resume=args.resume)


if __name__ == "__main__":
    main()