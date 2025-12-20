# app/backend/services/inference.py

import torch
import torch.nn as nn
from torchvision import models

from core.config import settings
from services.gradcam import GradCAM
from utils.image_ops import build_transform


class ResNet18WithHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        head = nn.Linear(512, num_classes)
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


class InferenceService:
    def __init__(self):
        self.device = torch.device(settings.device)  # cpu-only by design
        self.model, self.class_to_idx, self.idx_to_class = self._load_model(settings.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Best Grad-CAM target: last conv in layer4[-1]
        target_layer = self.model.backbone.layer4[-1].conv2
        self.gradcam = GradCAM(self.model, target_layer)

        self.tfm = build_transform(settings.image_size)

    def _load_model(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        class_to_idx = ckpt["class_to_idx"]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)

        model = ResNet18WithHead(num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, class_to_idx, idx_to_class

    def predict_with_gradcam(self, rgb_img):
        x = self.tfm(rgb_img).unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(True):
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            pred_idx = int(pred_idx.item())
            confidence = float(conf.item())
            pred_class = self.idx_to_class[pred_idx]

            score = logits[0, pred_idx]
            cam = self.gradcam.generate(score)

        return pred_class, confidence, cam
