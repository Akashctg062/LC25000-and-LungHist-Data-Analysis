"""
models.py
=========
Defines all 12 model architectures used in the LC25000 study and provides
helper functions to load trained weights and preprocess images.

This file is imported by app.py. It contains NO Streamlit code so it can be
unit-tested or reused independently.

IMPORTANT: every model here is built with `pretrained=False` / `weights=None`.
We do NOT want to download ImageNet weights — we immediately overwrite all
weights with your own trained checkpoint via `load_state_dict`. Using
pretrained=False makes the app start faster and work fully offline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm

# ----------------------------------------------------------------------
# Constants — these MUST match the values used during training
# ----------------------------------------------------------------------
NUM_CLASSES   = 5
IMG_SIZE      = 224
CLASS_NAMES   = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

# Human-readable labels for display in the UI
CLASS_LABELS = {
    'colon_aca': 'Colon Adenocarcinoma',
    'colon_n':   'Colon Benign Tissue',
    'lung_aca':  'Lung Adenocarcinoma',
    'lung_n':    'Lung Benign Tissue',
    'lung_scc':  'Lung Squamous Cell Carcinoma',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------
# Image preprocessing — MUST be identical to the eval_transform used
# during training and testing, otherwise predictions will be wrong.
# ----------------------------------------------------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ======================================================================
# CUSTOM ARCHITECTURE: FSPAN-Y
# (existing hybrid from the study — ConvNeXt-Tiny + attention + Y-Blocks)
# ======================================================================
class YBlock(nn.Module):
    """Dilated dual-path block with residual addition."""
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        a = F.relu(self.bn1(self.conv1(x)), inplace=True)
        b = F.relu(self.bn2(self.conv2(a)), inplace=True)
        return F.relu(a + b, inplace=True)


class FSPAN_Y(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny', pretrained=False, features_only=True, out_indices=(3,)
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        feat_ch = self.backbone.feature_info.channels()[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feat_ch, 512, 1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 16, 1),  nn.ReLU(),
            nn.Conv2d(16, 1, 1),   nn.Sigmoid(),
        )
        self.y1 = YBlock(512, 256, dilation=2)
        self.y2 = YBlock(256, 128, dilation=2)
        self.y3 = YBlock(128, 64,  dilation=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        f = self.backbone(x)[0]
        f = self.bottleneck(f)
        attn = self.attention(f)
        f = f * attn
        f = self.y1(f)
        f = self.y2(f)
        f = self.y3(f)
        f = self.gap(f).flatten(1)
        f = self.dropout(f)
        return self.fc(f)


# ======================================================================
# CUSTOM ARCHITECTURE: DPCT-Net
# (proposed novel — dual frozen backbones + gated fusion)
# ======================================================================
class DPCT_Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = timm.create_model('convnext_tiny', pretrained=False,
                                     num_classes=0, global_pool='avg')
        self.tfm = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False,
                                     num_classes=0, global_pool='avg')
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.tfm.parameters(): p.requires_grad = False
        cnn_dim = self.cnn.num_features
        tfm_dim = self.tfm.num_features
        d = 256
        self.proj_cnn = nn.Sequential(nn.Linear(cnn_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.proj_tfm = nn.Sequential(nn.Linear(tfm_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.cross_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(),
            nn.Linear(d, 2), nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, num_classes))

    def forward(self, x, return_gate=False):
        with torch.no_grad():
            f_cnn = self.cnn(x)
            f_tfm = self.tfm(x)
        z_cnn = self.proj_cnn(f_cnn)
        z_tfm = self.proj_tfm(f_tfm)
        seq = torch.stack([z_cnn, z_tfm], dim=1)
        attn_out, _ = self.cross_attn(seq, seq, seq)
        gate = self.gate(torch.cat([z_cnn, z_tfm], dim=-1))
        f_fused = gate[:, 0:1] * attn_out[:, 0] + gate[:, 1:2] * attn_out[:, 1]
        if return_gate:
            return self.classifier(f_fused), gate
        return self.classifier(f_fused)


# ======================================================================
# CUSTOM ARCHITECTURE: MSCA-Net
# (proposed novel — multi-scale cascade attention)
# ======================================================================
class MSCA_Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny', pretrained=False,
            features_only=True, out_indices=(1, 2, 3)
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        channels = self.backbone.feature_info.channels()
        d = 256
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, d, 1), nn.BatchNorm2d(d), nn.GELU())
            for c in channels
        ])
        self.ca = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(d, d // 8, 1), nn.ReLU(inplace=True),
                nn.Conv2d(d // 8, d, 1), nn.Sigmoid()
            ) for _ in channels
        ])
        self.spatial_attn = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, 1, 1), nn.Sigmoid())
            for _ in channels
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale_fusion = nn.Sequential(
            nn.Linear(d * len(channels), d), nn.GELU(), nn.LayerNorm(d)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, num_classes))

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        projected = [proj(f) for proj, f in zip(self.proj, feats)]
        refined = [None] * len(projected)
        refined[-1] = projected[-1] * self.ca[-1](projected[-1])
        for i in range(len(projected) - 2, -1, -1):
            attn_deep = self.spatial_attn[i + 1](refined[i + 1])
            attn_deep = F.interpolate(attn_deep, size=projected[i].shape[2:],
                                      mode='bilinear', align_corners=False)
            refined[i] = projected[i] * self.ca[i](projected[i]) * attn_deep
        pooled = [self.gap(r).flatten(1) for r in refined]
        fused = torch.cat(pooled, dim=1)
        z = self.scale_fusion(fused)
        return self.classifier(z)


# ======================================================================
# BUILD FUNCTIONS — one per model. Each returns an untrained nn.Module
# with the correct architecture. Weights are loaded separately.
# ======================================================================
def build_vgg16(num_classes=NUM_CLASSES):
    model = models.vgg16(weights=None)
    in_f = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_f, num_classes)
    return model

def build_resnet50(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

def build_mobilenetv3(num_classes=NUM_CLASSES):
    return timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=num_classes)

def build_vit(num_classes=NUM_CLASSES):
    return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)

def build_convnext(num_classes=NUM_CLASSES):
    return timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)

def build_efficientnet_b0(num_classes=NUM_CLASSES):
    return timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)

def build_efficientnet_b3(num_classes=NUM_CLASSES):
    return timm.create_model('tf_efficientnet_b3', pretrained=False, num_classes=num_classes)

def build_swin_tiny(num_classes=NUM_CLASSES):
    return timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)

def build_densenet121(num_classes=NUM_CLASSES):
    return timm.create_model('densenet121', pretrained=False, num_classes=num_classes)

def build_fspan_y(num_classes=NUM_CLASSES):
    return FSPAN_Y(num_classes=num_classes)

def build_dpct(num_classes=NUM_CLASSES):
    return DPCT_Net(num_classes=num_classes)

def build_msca(num_classes=NUM_CLASSES):
    return MSCA_Net(num_classes=num_classes)


# ======================================================================
# MODEL REGISTRY — maps the checkpoint name to its build function.
# The keys MUST match your saved checkpoint filenames:
#   "{key}_best.pth"  e.g.  "VGG16_best.pth"
# ======================================================================
MODEL_REGISTRY = {
    "VGG16":           build_vgg16,
    "ResNet50":        build_resnet50,
    "MobileNetV3":     build_mobilenetv3,
    "ViT_Base":        build_vit,
    "ConvNeXt_Tiny":   build_convnext,
    "FSPAN_Y":         build_fspan_y,
    "EfficientNet_B0": build_efficientnet_b0,
    "DenseNet121":     build_densenet121,
    "MSCA_Net":        build_msca,
    "EfficientNet_B3": build_efficientnet_b3,
    "Swin_Tiny":       build_swin_tiny,
    "DPCT_Net":        build_dpct,
}

# Friendly metadata for display
MODEL_FAMILY = {
    "VGG16":           "Classic CNN",
    "ResNet50":        "Classic CNN",
    "DenseNet121":     "Classic CNN",
    "MobileNetV3":     "Efficient CNN",
    "EfficientNet_B0": "Efficient CNN",
    "EfficientNet_B3": "Efficient CNN",
    "ConvNeXt_Tiny":   "Modern CNN",
    "ViT_Base":        "Transformer",
    "Swin_Tiny":       "Transformer",
    "FSPAN_Y":         "Hybrid",
    "DPCT_Net":        "Hybrid (proposed)",
    "MSCA_Net":        "Hybrid (proposed)",
}


# ======================================================================
# LOADING — build architecture, load trained weights, set to eval mode
# ======================================================================
def load_model(model_name, weights_path, device):
    """
    Build the architecture for `model_name`, load the trained state_dict
    from `weights_path`, move it to `device`, and put it in eval mode.

    Returns the ready-to-use model.
    Raises FileNotFoundError if the checkpoint does not exist.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    # 1. Build the (untrained) architecture
    model = MODEL_REGISTRY[model_name]()

    # 2. Load the trained weights. weights_only=True is safe here because
    #    our checkpoints are plain state_dicts (tensors only).
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 3. Move to device and switch to eval mode (disables dropout, etc.)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_tensor, device):
    """
    Run a single forward pass and return softmax probabilities.

    image_tensor: a single preprocessed image tensor of shape (3, 224, 224)
    Returns: a 1-D numpy array of 5 probabilities (one per class).
    """
    x = image_tensor.unsqueeze(0).to(device)   # add batch dim -> (1, 3, 224, 224)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).cpu().numpy()
