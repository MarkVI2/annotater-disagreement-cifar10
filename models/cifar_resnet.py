import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file
import os

class CIFAR10ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64 # number of input channels

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,2,stride=1)
        self.layer2 = self._make_layer(128,2,stride=1)
        self.layer3 = self._make_layer(256,2,stride=1)
        self.layer4 = self._make_layer(512,2,stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1) # init weights to 1
                nn.init.constant_(m.bias, 0) # init bias to 0

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0) # init tensor to a specific constant value
    
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

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential() # skip connection
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class SoftLabelPredictor(nn.Module):
    def __init__(self, backbone, num_classes=10, head_type='Linear'):
        super().__init__()
        self.backbone = backbone
        self.backbone_out = 512

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
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return F.softmax(logits, dim=1)
    
LOCAL_WEIGHTS_DEFAULT = os.path.join('models', 'weights', 'resnet18.pt')

def load_local_cifar10_weights(weights_path, map_location='cpu'):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}. "
            "Please place the .pth from the Google Drive zip into that location.")
    state = torch.load(weights_path, map_location=map_location)
    return state

def build_soft_label_model(head_type='linear', pretrained=True, weights_path=None, download_fallback=False):
    backbone = CIFAR10ResNet18Backbone()

    if pretrained:
        path = weights_path or LOCAL_WEIGHTS_DEFAULT
        try:
            state = load_local_cifar10_weights(path)
            print(f"Loaded backbone weights form {path}")
        except FileNotFoundError:
            if download_fallback:
                print(f"Local weights not found at {path}, downloading from kaggle")
