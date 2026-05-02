import torch
from data.pipeline import load_full_dataset, create_splits, create_dataloaders, get_transforms
from training.config import FIXED_SEED, DATA_DIR, CIFAR10H_PROBS_FILE, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO
from models.cifar_resnet import build_soft_label_model

model = build_soft_label_model(pretrained=False)
ckpt = torch.load('outputs/checkpoints/kl_linear/best.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
# Use device detection to avoid failing if CUDA is absent.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

dataset = load_full_dataset(DATA_DIR, CIFAR10H_PROBS_FILE)
splits = create_splits(dataset, TRAIN_RATIO, VAL_RATIO, FIXED_SEED)
transforms_dict = {
    'train': get_transforms(True),
    'val': get_transforms(False),
    'test': get_transforms(False)
}
loaders = create_dataloaders(splits, BATCH_SIZE, 0, transforms_dict)

total_ce = 0
with torch.no_grad():
    for img, tgt, _ in loaders['val']:
        pred = model(img.to(device)).cpu()
        ce = -(tgt * (pred + 1e-9).log()).sum(dim=1).mean()
        total_ce += ce.item()
print(f"Cross-entropy (nats): {total_ce/len(loaders['val']):.4f}")
