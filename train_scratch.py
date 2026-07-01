import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet_cifar import resnet18_cifar
from utils.data import build_loaders
from utils.engine import set_seed, run_one_epoch, write_csv_header, append_csv, save_checkpoint
from utils.plot import plot_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-18 CIFAR variant from scratch on CIFAR-10")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./outputs/scratch")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_loaders(args.data_dir, args.batch_size, args.num_workers, args.seed)
    model = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = out_dir / "history.csv"
    log_path = out_dir / "train.log"
    write_csv_header(csv_path)
    best_acc = 0.0

    with open(log_path, "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_one_epoch(model, val_loader, criterion, None, device, train=False)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            append_csv(csv_path, [epoch, train_loss, train_acc, val_loss, val_acc, lr])
            msg = f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.6f}"
            print(msg)
            log.write(msg + "\n")
            log.flush()
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(out_dir / "best_model.pth", model, optimizer, scheduler, epoch, best_acc, args)

    plot_curves(str(csv_path), str(out_dir))
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
