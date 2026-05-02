import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.models.resnet import resnet18 as torchvision_resnet18

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CIFAR10ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1)
        self.layer3 = self._make_layer(256, 2, stride=1)
        self.layer4 = self._make_layer(512, 2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

class SoftLabelPredictor(nn.Module):
    def __init__(self, backbone, num_classes=10, head_type='linear', use_temperature=False, init_temp=2.0):
        super().__init__()
        self.backbone = backbone
        self.backbone_out = 512
        self.use_temperature = use_temperature
        self.head_type = head_type

        if head_type == 'linear':
            self.head = nn.Linear(self.backbone_out, num_classes)
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.backbone_out, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"head_type must be 'linear' or 'mlp', got {head_type}")

        if use_temperature:
            self.temperature = nn.Parameter(torch.tensor(init_temp))
        else:
            self.temperature = None

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        if self.use_temperature:
            logits = logits / self.temperature
        return F.softmax(logits, dim=1)

DEFAULT_WEIGHTS_PATH = os.path.join('models', 'weights', 'resnet18.pt')

def load_cifar10_pretrained(backbone, weights_path):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}.")
    state = torch.load(weights_path, map_location='cpu')
    msg = backbone.load_state_dict(state, strict=False)
    print(f"Loaded CIFAR-10 backbone weights from {weights_path}")
    print(f"  Missing keys (expected): {msg.missing_keys}")
    print(f"  Unexpected keys: {msg.unexpected_keys}")
    return backbone

def load_imagenet_pretrained(backbone):
    # Load torchvision ResNet‑18 pretrained on ImageNet
    pretrained_model = torchvision_resnet18(pretrained=True)
    target_state = backbone.state_dict()
    pretrained_state = pretrained_model.state_dict()

    # Keys that can be directly copied (everything except conv1)
    for k in target_state:
        if k == 'conv1.weight':
            # Adapt 7x7 -> 3x3: average the 7x7 kernel to a 3x3 kernel (a simple approach)
            # or just leave our conv1 random. We'll leave it random and only copy the rest.
            continue
        if k in pretrained_state and target_state[k].shape == pretrained_state[k].shape:
            target_state[k] = pretrained_state[k]
    backbone.load_state_dict(target_state, strict=False)
    print("Loaded ImageNet pretrained backbone (except conv1).")
    return backbone

def build_soft_label_model(head_type='linear', pretrained_type='random',
                           weights_path=None, use_temperature=False, init_temp=2.0):
    """
    pretrained_type: 'random', 'cifar10', 'imagenet'
    """
    backbone = CIFAR10ResNet18Backbone()

    if pretrained_type == 'cifar10':
        path = weights_path or DEFAULT_WEIGHTS_PATH
        load_cifar10_pretrained(backbone, path)
    elif pretrained_type == 'imagenet':
        load_imagenet_pretrained(backbone)
    elif pretrained_type == 'random':
        pass  # already random
    else:
        raise ValueError(f"Unknown pretrained_type: {pretrained_type}")

    model = SoftLabelPredictor(backbone, num_classes=10, head_type=head_type,
                               use_temperature=use_temperature, init_temp=init_temp)
    return model