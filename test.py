import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet18

from models.resnet_cifar import resnet18_cifar
from train_finetune import build_model as build_finetune_model
from utils.data import build_loaders
from utils.engine import set_seed, run_one_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CIFAR-10 ResNet checkpoint")
    p.add_argument("--mode", choices=["scratch", "finetune"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers, args.seed)

    if args.mode == "scratch":
        model = resnet18_cifar(num_classes=10)
    else:
        model = build_finetune_model(freeze_backbone=False)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_one_epoch(model, test_loader, criterion, None, device, train=False)

    out_dir = Path(args.ckpt).parent
    report = f"mode={args.mode}\ncheckpoint={args.ckpt}\ntest_loss={test_loss:.6f}\ntest_accuracy={test_acc:.6f}\n"
    print(report)
    with open(out_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
