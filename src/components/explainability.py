# src/components/explainability.py

import os
import sys
import json
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import matplotlib.pyplot as plt

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import ExplainabilityConfig
from src.entity.artifact_entity import (
    DataSplittingArtifact,
    ModelTrainerArtifact,
    ExplainabilityArtifact
)
from src.components.data_loader import get_device

from src.utils.mlflow_utils import (
    setup_mlflow,
    start_run,
    log_params,
    log_metrics,
    log_artifacts,
    end_run
)
from src.entity.config_entity import MLflowConfig


# -----------------------------
# Utilities
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def unnormalize(t: torch.Tensor) -> torch.Tensor:
    """
    t: [3, H, W] normalized tensor -> [0,1] tensor
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(3, 1, 1)
    x = t * std + mean
    return torch.clamp(x, 0.0, 1.0)


def softmax_confidence(logits: torch.Tensor) -> Tuple[int, float]:
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf.item())


# -----------------------------
# Model (same as fine-tune)
# -----------------------------
class ResNet18WithHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        head = nn.Linear(512, num_classes)
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple; grad_out[0] is gradients w.r.t. layer output
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, class_score: torch.Tensor) -> torch.Tensor:
        """
        class_score: scalar tensor (logit for target class)
        Returns: CAM [H, W] normalized to [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        class_score.backward(retain_graph=True)

        # activations: [B, C, H, W], gradients: [B, C, H, W]
        grads = self.gradients[0]      # [C, H, W]
        acts = self.activations[0]     # [C, H, W]

        # weights: global avg pool over H,W
        weights = torch.mean(grads, dim=(1, 2))  # [C]

        cam = torch.zeros(acts.shape[1:], device=acts.device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


class Explainability:
    def __init__(self, explain_config: ExplainabilityConfig):
        self.config = explain_config
        self.device = get_device()

    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _collect_samples(self, test_dir: str, samples_per_class: int) -> List[Tuple[str, str]]:
        """
        Returns list of (image_path, true_class_name)
        """
        pairs = []
        classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

        for cls in classes:
            cls_dir = os.path.join(test_dir, cls)
            imgs = sorted([os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
            selected = imgs[:samples_per_class]
            for p in selected:
                pairs.append((p, cls))

        if not pairs:
            raise ValueError(f"No images found under test_dir: {test_dir}")

        return pairs

    def _save_overlay(
        self,
        rgb_img: Image.Image,
        cam: torch.Tensor,
        out_path: str,
        title: str
    ) -> None:
        """
        rgb_img: PIL image (original)
        cam: [H,W] tensor in [0,1]
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cam_np = cam.detach().cpu().numpy()
        img_np = np.array(rgb_img).astype(np.float32) / 255.0

        # Resize CAM to image size if needed
        if cam_np.shape[0] != img_np.shape[0] or cam_np.shape[1] != img_np.shape[1]:
            cam_img = Image.fromarray((cam_np * 255).astype(np.uint8))
            cam_img = cam_img.resize((img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR)
            cam_np = np.array(cam_img).astype(np.float32) / 255.0

        fig = plt.figure()
        plt.imshow(img_np)
        plt.imshow(cam_np, alpha=0.45)  # overlay with default colormap
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    def initiate_explainability(
        self,
        data_splitting_artifact: DataSplittingArtifact,
        finetune_artifact: ModelTrainerArtifact,
    ) -> ExplainabilityArtifact:
        try:
            logging.info(f"[Grad-CAM] Using device: {self.device}")
            
            mlflow_cfg = MLflowConfig()
            setup_mlflow(mlflow_cfg.experiment_name, mlflow_cfg.tracking_uri)

            run = start_run(
                run_name="gradcam_explainability",
                tags={"stage": "explainability", "method": "grad-cam"}
            )

            # Prepare output dirs
            correct_dir = os.path.join(self.config.heatmaps_dir, "correct")
            incorrect_dir = os.path.join(self.config.heatmaps_dir, "incorrect")
            os.makedirs(correct_dir, exist_ok=True)
            os.makedirs(incorrect_dir, exist_ok=True)

            # Load trained checkpoint (from fine-tune stage)
            ckpt = torch.load(finetune_artifact.trained_model_path, map_location="cpu")
            class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            num_classes = len(class_to_idx)

            model = ResNet18WithHead(num_classes=num_classes)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(self.device)
            model.eval()

            # Grad-CAM target layer: ResNet18 last conv block
            target_layer = model.backbone.layer4
            cam_engine = GradCAM(model, target_layer)

            # Collect samples
            samples = self._collect_samples(
                test_dir=data_splitting_artifact.test_dir,
                samples_per_class=self.config.samples_per_class
            )
            
            log_params({
                "samples_per_class": self.config.samples_per_class,
                "total_samples": len(samples),
                "device": str(self.device),
            })

            tfm = get_eval_transform(224)

            index_entries = []
            total = 0

            for img_path, true_class in samples:
                rgb = Image.open(img_path).convert("RGB")
                x = tfm(rgb).unsqueeze(0).to(self.device)

                with torch.set_grad_enabled(True):
                    logits = model(x)
                    pred_idx, conf = softmax_confidence(logits)

                    true_idx = class_to_idx[true_class]
                    pred_class = idx_to_class[pred_idx]
                    is_correct = (pred_idx == true_idx)

                    # Grad-CAM w.r.t predicted class (most interpretable for "why predicted")
                    score = logits[0, pred_idx]
                    cam = cam_engine.generate(score)

                out_dir = correct_dir if is_correct else incorrect_dir
                base = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"{base}__true_{true_class}__pred_{pred_class}__conf_{conf:.3f}.png"
                out_path = os.path.join(out_dir, out_name)

                title = f"True: {true_class} | Pred: {pred_class} | Conf: {conf:.3f}"
                self._save_overlay(rgb, cam, out_path, title)

                index_entries.append({
                    "image_path": img_path,
                    "output_path": out_path,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "confidence": conf,
                    "correct": is_correct
                })

                total += 1
                if total % 10 == 0:
                    logging.info(f"[Grad-CAM] Processed {total}/{len(samples)}")

            cam_engine.remove()

            index_payload = {
                "device": str(self.device),
                "samples_per_class": self.config.samples_per_class,
                "total_samples": total,
                "heatmaps_dir": self.config.heatmaps_dir,
                "entries": index_entries
            }
            self._write_json(self.config.index_file_path, index_payload)

            log_metrics({
                "total_samples": total
            })

            # Log entire explainability folder (heatmaps + index.json)
            log_artifacts(self.config.explainability_dir)

            end_run()

            logging.info("[Grad-CAM] Explainability generation completed successfully.")

            return ExplainabilityArtifact(
                explainability_dir=self.config.explainability_dir,
                heatmaps_dir=self.config.heatmaps_dir,
                index_file_path=self.config.index_file_path,
                total_samples=total
            )

        except Exception as e:
            raise CustomException(e, sys)
