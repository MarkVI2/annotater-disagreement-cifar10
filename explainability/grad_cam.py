import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image: torch.Tensor, target_class=None, strategy="top_pred"):
        self.model.eval()
        image = image.requires_grad_(True)
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)

        if target_class is not None:
            score = probs[0, target_class]
        elif strategy == "top_pred":
            score = probs.max(dim=1)[0]  # just a scalar
        elif strategy == "entropy_weighted":
            eps = 1e-12
            weights = -(probs * torch.log(probs + eps))
            score = (weights * probs).sum()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.model.zero_grad()
        score.backward()
        cam = self._compute_cam()
        return cam

    def _compute_cam(self):
        grads = self._gradients[0]  # (C, H, W)
        acts  = self._activations[0] # (C, H, W)
        weights = grads.mean(dim=(1,2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @staticmethod
    def overlay(raw_image: np.ndarray, cam: np.ndarray, alpha=0.45):
        h, w = raw_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        img_float = raw_image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float /= 255.0
        return np.clip((1 - alpha) * img_float + alpha * heatmap, 0, 1)