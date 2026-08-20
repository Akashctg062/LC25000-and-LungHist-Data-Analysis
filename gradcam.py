"""
gradcam.py
==========
A self-contained Grad-CAM implementation (no external grad-cam library needed).

Grad-CAM produces a heatmap showing which regions of the input image most
influenced the model's prediction. It works by:
  1. Capturing the activations of a chosen convolutional layer (forward hook).
  2. Capturing the gradients flowing into that layer (backward hook).
  3. Weighting the activation channels by the average gradient, summing them,
     and applying ReLU to keep only positive contributions.

Grad-CAM needs a spatial feature map (channels x height x width) that has
gradients. This works for all the convolutional models. It does NOT work for:
  - ViT_Base, Swin_Tiny : pure transformers, no good conv feature map
  - DPCT_Net            : its backbones run inside torch.no_grad(), so no
                          spatial gradients are available
For those three, get_target_layer() returns None and the UI shows a note.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image

from models import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


# ----------------------------------------------------------------------
# Choose which layer to attach Grad-CAM hooks to, for each architecture.
# Returns None if Grad-CAM is not supported for that model.
# ----------------------------------------------------------------------
def get_target_layer(model, model_name):
    try:
        if model_name == "VGG16":
            # last conv layer inside the feature extractor
            convs = [m for m in model.features if m.__class__.__name__ == "Conv2d"]
            return convs[-1]
        if model_name == "ResNet50":
            return model.layer4[-1]                  # last residual block
        if model_name == "DenseNet121":
            return model.features.norm5              # final spatial feature map
        if model_name == "MobileNetV3":
            return model.conv_head                   # final 1x1 conv
        if model_name in ("EfficientNet_B0", "EfficientNet_B3"):
            return model.conv_head                   # final 1x1 conv
        if model_name == "ConvNeXt_Tiny":
            return model.stages[-1]                  # last ConvNeXt stage
        if model_name == "FSPAN_Y":
            return model.y3                          # last Y-Block (trainable, spatial)
        if model_name == "MSCA_Net":
            return model.proj[-1]                    # deepest-scale projection (spatial)
        # ViT_Base, Swin_Tiny, DPCT_Net -> not supported
        return None
    except Exception:
        return None


GRADCAM_SUPPORTED = {
    "VGG16", "ResNet50", "DenseNet121", "MobileNetV3",
    "EfficientNet_B0", "EfficientNet_B3", "ConvNeXt_Tiny",
    "FSPAN_Y", "MSCA_Net",
}
GRADCAM_UNSUPPORTED_REASON = {
    "ViT_Base":  "Pure transformer — no convolutional feature map for Grad-CAM.",
    "Swin_Tiny": "Pure transformer — no convolutional feature map for Grad-CAM.",
    "DPCT_Net":  "Frozen dual backbones run without gradients — Grad-CAM not available.",
}


class GradCAM:
    """
    Minimal Grad-CAM. Usage:
        gc = GradCAM(model, target_layer)
        cam = gc(input_tensor, class_idx)   # returns HxW heatmap in [0,1]
        gc.remove_hooks()

    Implementation note: we use a forward hook to grab the layer's output
    activations, and then register a hook ON THAT OUTPUT TENSOR to grab its
    gradient during backward. This is more robust than
    `register_full_backward_hook`, which conflicts with the in-place ReLU
    operations used inside VGG16, MobileNetV3, and EfficientNet.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # Save the activations for later
        self.activations = out
        # Register a hook on the output tensor itself to capture its gradient.
        # This only works if `out` participates in the autograd graph.
        if out.requires_grad:
            out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def __call__(self, input_tensor, class_idx):
        """
        input_tensor : (1, 3, 224, 224) on the correct device, requires_grad=True
        class_idx    : which class score to back-propagate from
        returns      : a (H, W) numpy heatmap normalized to [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)            # forward pass (forward hook fires)
        score = logits[0, class_idx]                 # scalar score for target class
        score.backward()                             # backward pass (tensor hook fires)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        # activations & gradients: (1, C, h, w)
        grads = self.gradients[0]                    # (C, h, w)
        acts  = self.activations[0].detach()         # (C, h, w)
        weights = grads.mean(dim=(1, 2))             # (C,) — global-average-pool the grads

        # Weighted combination of activation maps: cam = sum_c (w_c * A_c)
        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)
        cam = F.relu(cam)                            # keep only positive influence

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()

    def remove_hooks(self):
        self._fwd.remove()


def tensor_to_display_image(image_tensor):
    """Undo ImageNet normalization so we can show the original image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (image_tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_heatmap(image_tensor, cam, alpha=0.5):
    """
    Overlay a Grad-CAM heatmap on the original image.

    image_tensor : the preprocessed (3, 224, 224) tensor
    cam          : (h, w) heatmap in [0, 1] from GradCAM.__call__
    alpha        : blend factor (0 = only image, 1 = only heatmap)
    Returns a PIL.Image ready to display.
    """
    base = tensor_to_display_image(image_tensor)                 # (224, 224, 3) uint8

    # Resize the (small) CAM up to the image size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    cam_resized = np.array(cam_img) / 255.0

    # Apply the 'jet' colormap to turn the heatmap into RGB
    colored = cm.jet(cam_resized)[:, :, :3]                      # (224, 224, 3) float [0,1]
    colored = (colored * 255).astype(np.uint8)

    # Blend
    blended = (alpha * colored + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)


def compute_gradcam(model, model_name, image_tensor, class_idx, device):
    """
    High-level helper: returns a PIL overlay image, or None if unsupported.
    """
    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        return None
    gc = GradCAM(model, target_layer)
    try:
        x = image_tensor.unsqueeze(0).to(device)
        x.requires_grad_(True)
        cam = gc(x, class_idx)
        overlay = overlay_heatmap(image_tensor, cam, alpha=0.5)
    finally:
        gc.remove_hooks()
    return overlay
