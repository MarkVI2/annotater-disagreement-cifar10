# training/train.py
import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

from data.pipeline import load_full_dataset, create_splits, create_dataloaders, get_transforms
from training.config import (
    FIXED_SEED, DATA_DIR, CIFAR10H_PROBS_FILE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_RATIO, VAL_RATIO, CHECKPOINTS_DIR, LOGS_DIR
)
from training.metrics_logger import MetricsLogger
from models.cifar_resnet import build_soft_label_model
from utils.device import get_device, get_autocast_context
from losses import get_loss_function      # we'll build this

def train_one_epoch(model, loader, optimizer, device, scaler, autocast_ctx, loss_fn):
    model.train()
    running_loss = 0.0
    for images, soft_targets, _ in tqdm.tqdm(loader, desc='Training', leave=False):
        images = images.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True)

        with autocast_ctx:
            pred_probs = model(images)     # already softmaxed
            loss = loss_fn(pred_probs, soft_targets)

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

def validate(model, loader, device, loss_fn):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, soft_targets, _ in tqdm.tqdm(loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            pred_probs = model(images)
            loss = loss_fn(pred_probs, soft_targets)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default=CHECKPOINTS_DIR)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--loss', type=str, default='kl', choices=['kl', 'js', 'emd', 'custom_composite'])
    parser.add_argument('--loss_beta', type=float, default=0.5, help='Weight for entropy penalty in custom composite loss')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # ---- Data pipeline ----
    # Use your teammate's pipeline directly
    dataset = load_full_dataset(DATA_DIR, CIFAR10H_PROBS_FILE)
    splits = create_splits(dataset, TRAIN_RATIO, VAL_RATIO, FIXED_SEED)

    # Transforms
    transforms_dict = {
        'train': get_transforms(is_train=True),
        'val': get_transforms(is_train=False),
        'test': get_transforms(is_train=False),
    }

    loaders = create_dataloaders(splits, args.batch_size, NUM_WORKERS, transforms_dict)
    train_loader = loaders['train']
    val_loader = loaders['val']

    # ---- Model ----
    model = build_soft_label_model(head_type=args.head, pretrained=args.pretrained)
    model = model.to(device)

    # ---- Loss function ----
    loss_fn = get_loss_function(args.loss, beta=args.loss_beta)
    print(f"Using loss: {args.loss}")

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Mixed precision ----
    use_amp = not args.no_amp and device.type in ('cuda', 'mps')
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
    autocast_ctx = get_autocast_context(device) if use_amp else torch.no_grad()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    logger = MetricsLogger(LOGS_DIR, filename=f'{args.loss}_metrics.csv')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, autocast_ctx, loss_fn)
        val_loss = validate(model, val_loader, device, loss_fn)
        scheduler.step()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:03d} | Train {args.loss}: {train_loss:.4f} | Val {args.loss}: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        logger.log(epoch, {'train_loss': train_loss, 'val_loss': val_loss})

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

    logger.close()
    print(f"Training finished. Best val {args.loss}: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()