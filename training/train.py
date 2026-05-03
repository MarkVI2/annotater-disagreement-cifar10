import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm
import json
import numpy as np

from data.pipeline import load_full_dataset, create_splits, create_dataloaders, get_transforms
from training.config import (
    FIXED_SEED, DATA_DIR, CIFAR10H_PROBS_FILE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_RATIO, VAL_RATIO, CHECKPOINTS_DIR, LOGS_DIR
)
from training.metrics_logger import MetricsLogger
from models.cifar_resnet import build_soft_label_model
from utils.device import get_device, get_autocast_context
from losses import get_loss_function


def train_one_epoch(model, loader, optimizer, device, scaler, autocast_ctx, loss_fn):
    model.train()
    running_loss = 0.0
    for images, soft_targets, _ in tqdm.tqdm(loader, desc='Training', leave=False):
        images = images.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True)

        with autocast_ctx:
            pred_probs = model(images)
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
    parser.add_argument('--pretrained_type', type=str, default='cifar10',
                        choices=['random', 'cifar10', 'imagenet'])
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--loss', type=str, default='kl',
                        choices=['kl', 'js', 'emd', 'custom_composite'])
    parser.add_argument('--loss_beta', type=float, default=0.5,
                        help='Weight for entropy penalty in custom composite loss')
    parser.add_argument('--loss_epsilon', type=float, default=0.1,
                        help='Epsilon for EMD loss')
    parser.add_argument('--use_temperature', action='store_true',
                        help='Use learnable temperature scaling head')
    parser.add_argument('--init_temp', type=float, default=2.0,
                        help='Initial temperature value')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Data pipeline
    dataset = load_full_dataset(DATA_DIR, CIFAR10H_PROBS_FILE)
    splits = create_splits(dataset, TRAIN_RATIO, VAL_RATIO, FIXED_SEED)
    transforms_dict = {
        'train': get_transforms(is_train=True),
        'val': get_transforms(is_train=False),
        'test': get_transforms(is_train=False),
    }
    loaders = create_dataloaders(splits, args.batch_size, NUM_WORKERS, transforms_dict)
    train_loader = loaders['train']
    val_loader = loaders['val']

    model = build_soft_label_model(
        head_type=args.head,
        pretrained_type=args.pretrained_type,
        use_temperature=args.use_temperature,
        init_temp=args.init_temp
    )
    model = model.to(device)
    model = torch.compile(model, mode='reduce-overhead')

    # Loss
    loss_fn = get_loss_function(args.loss, beta=args.loss_beta, epsilon=args.loss_epsilon)
    print(f"Using loss: {args.loss}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision
    use_amp = not args.no_amp and device.type in ('cuda', 'mps')
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
    autocast_ctx = get_autocast_context(device) if use_amp else torch.no_grad()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    pretrain_tag = f"pt_{args.pretrained_type}"
    temp_tag = f"_temp{args.init_temp}" if args.use_temperature else ""
    exp_name = f'{args.loss}_{args.head}_{pretrain_tag}{temp_tag}'
    logger = MetricsLogger(LOGS_DIR, filename_prefix=exp_name)

    hyperparams = vars(args)
    with open(os.path.join(args.save_dir, f'{exp_name}_hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, autocast_ctx, loss_fn)
        val_loss = validate(model, val_loader, device, loss_fn)
        scheduler.step()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:03d} | Train {args.loss}: {train_loss:.4f} | Val {args.loss}: {val_loss:.4f} | Time: {epoch_time:.1f}s")

        logger.log(epoch, {'train_loss': train_loss, 'val_loss': val_loss})

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'{exp_name}_latest.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, f'{exp_name}_best.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement in {args.patience} epochs).")
            break

    logger.close()
    print(f"Training finished. Best val {args.loss}: {best_val_loss:.4f}")

    # Generate training curves after training
    try:
        from visualisations.training_curves import plot_training_curves
        csv_path = os.path.join(LOGS_DIR, f"{exp_name}_metrics.csv")
        plot_training_curves(csv_path, label=exp_name)
    except Exception as e:
        print(f"Could not generate training curve plot: {e}")


if __name__ == '__main__':
    main()