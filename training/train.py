import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

from data.dataset import get_cifar10h_splits   # your teammate’s function
from models.cifar_resnet import build_soft_label_model
from utils.device import get_device, get_autocast_context

def train_one_epoch(model, loader, optimizer, device, scaler, autocast_ctx):
    model.train()
    running_loss = 0.0
    for images, target_dist in tqdm.tqdm(loader, desc='Training', leave=False):
        images = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)

        with autocast_ctx:
            pred_probs = model(images)        # already softmaxed
            log_probs = torch.log(pred_probs + 1e-9)
            loss = F.kl_div(log_probs, target_dist, reduction='batchmean')

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, target_dist in tqdm.tqdm(loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            target_dist = target_dist.to(device, non_blocking=True)
            pred_probs = model(images)
            log_probs = torch.log(pred_probs + 1e-9)
            loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Load CIFAR-10 backbone weights')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_set, val_set, test_set = get_cifar10h_splits()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    model = build_soft_label_model(head_type=args.head, pretrained=args.pretrained)
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision
    use_amp = not args.no_amp and device.type in ('cuda', 'mps')
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
    autocast_ctx = get_autocast_context(device) if use_amp else torch.no_grad()  # no‑op context

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, autocast_ctx)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:03d} | Train KL: {train_loss:.4f} | Val KL: {val_loss:.4f} | Time: {epoch_time:.1f}s")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'latest.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))

    print(f"Training finished. Best val KL: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()