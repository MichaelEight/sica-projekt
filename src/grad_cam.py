"""
1D Grad-CAM dla modelu Inception1D.
Generuje mapy ciepła wskazujące, które fragmenty sygnału
wpłynęły na predykcję danej klasy.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM1D:
    """
    Grad-CAM for 1D convolutional models.

    Hooks into the last convolutional layer of the backbone
    and computes gradient-weighted activation maps.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.feature_maps = None
        self.gradients = None

        # Hook into the last inception block
        last_block = model.backbone.blocks[-1]
        last_block.register_forward_hook(self._save_features)
        last_block.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def _forward_only(self, x):
        """Forward pass without grad tracking for getting logits."""
        return self.model(x)

    def generate(self, x, class_idx=None, signal_length=5000):
        """
        Generate Grad-CAM heatmap for a single input.

        Args:
            x: input tensor, shape (1, 12, 5000)
            class_idx: target class index (None = use predicted class)
            signal_length: original signal length for upsampling

        Returns:
            heatmap: numpy array of shape (signal_length,), values in [0, 1]
            class_idx: the class index used
            probability: sigmoid probability for that class
        """
        self.model.eval()

        # Need gradients for Grad-CAM
        x = x.clone().requires_grad_(True)

        # Forward pass
        logits = self.model(x)
        probs = torch.sigmoid(logits)

        if class_idx is None:
            class_idx = logits[0].argmax().item()

        probability = probs[0, class_idx].item()

        # Backward pass for target class
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, L)
        features = self.feature_maps[0]  # (C, L)

        # Global average pool the gradients (importance weights)
        weights = gradients.mean(dim=1)  # (C,)

        # Weighted combination of feature maps
        cam = torch.zeros(features.shape[1], device=features.device)
        for i, w in enumerate(weights):
            cam += w * features[i]

        # ReLU — only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to original signal length
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0).float()
        heatmap = F.interpolate(cam_tensor, size=signal_length, mode="linear", align_corners=False)
        heatmap = heatmap.squeeze().numpy()

        return heatmap, class_idx, probability

    def generate_all_classes(self, x, signal_length=5000):
        """
        Generate Grad-CAM heatmaps for all 8 classes.

        Args:
            x: input tensor, shape (1, 12, 5000)

        Returns:
            heatmaps: dict {class_idx: heatmap_array}
            probabilities: dict {class_idx: probability}
        """
        heatmaps = {}
        probabilities = {}

        for cls_idx in range(8):
            heatmap, _, prob = self.generate(x, class_idx=cls_idx, signal_length=signal_length)
            heatmaps[cls_idx] = heatmap
            probabilities[cls_idx] = prob

        return heatmaps, probabilities
