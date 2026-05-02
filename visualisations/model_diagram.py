import torch
from models.cifar_resnet import build_soft_label_model
from torchviz import make_dot

# Generate and save a diagram of the model
model = build_soft_label_model(head_type='linear', pretrained_type='random')
x = torch.randn(1, 3, 32, 32)
dot = make_dot(model(x), params=dict(model.named_parameters()))
dot.render('model_diagram', format='png', cleanup=True)
print("Model diagram saved as model_diagram.png")

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")