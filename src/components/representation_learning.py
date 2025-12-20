# src/components/representation_learning.py

import os
import sys
import json
from dataclasses import asdict
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import RepresentationLearningConfig
from src.entity.artifact_entity import RepresentationLearningArtifact
from src.components.data_loader import create_ssl_dataloader, get_device

from src.utils.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics, log_artifact, log_artifacts, end_run
from src.entity.config_entity import MLflowConfig

# ------------------------------
# Projection Head (SimCLR)
# ------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------
# NT-Xent Loss (InfoNCE)
# ------------------------------
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    z1, z2: [B, D]
    """
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T)      # [2B, 2B]

    # remove similarity with itself
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim[~mask].view(2 * batch_size, -1)  # [2B, 2B-1]

    sim = sim / temperature

    # positives: z1[i] with z2[i], and z2[i] with z1[i]
    pos = torch.sum(z1 * z2, dim=1) / temperature  # [B]
    pos = torch.cat([pos, pos], dim=0)             # [2B]

    # labels: positive pair is index 0 after we build logits
    logits = torch.cat([pos.unsqueeze(1), sim], dim=1)  # [2B, 1+(2B-1)]
    labels = torch.zeros(2 * batch_size, device=z.device, dtype=torch.long)

    return F.cross_entropy(logits, labels)


class RepresentationLearning:
    def __init__(self, representation_learning_config: RepresentationLearningConfig):
        self.config = representation_learning_config
        self.device = get_device()

    def _build_backbone(self) -> nn.Module:
        """
        ResNet-18 backbone without final classification layer.
        Output feature dim = 512.
        """
        model = models.resnet18(weights=None)  # SSL pretraining, no need for imagenet weights
        model.fc = nn.Identity()
        return model

    def _save_checkpoint(self, path: str, backbone: nn.Module, projector: nn.Module, epoch: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "projector_state_dict": projector.state_dict(),
            },
            path
        )

    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def initiate_representation_learning(self, ssl_train_dir: str) -> RepresentationLearningArtifact:
        """
        ssl_train_dir: typically Artifacts/<ts>/data_splitting/processed_split/train
        """
        try:
            logging.info(f"[SSL] Using device: {self.device}")

            mlflow_cfg = MLflowConfig()
            setup_mlflow("VisionOps", mlflow_cfg.tracking_uri)

            run = start_run(run_name="ssl_resnet18_simclr_lite", tags={"stage":"ssl"})
            log_params({
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.learning_rate,
            "temperature": self.config.temperature,
            "device": str(self.device),
            })
            
            # Data
            loader = create_ssl_dataloader(
                data_dir=ssl_train_dir,
                batch_size=self.config.batch_size,
                num_workers=2
            )

            # Model
            backbone = self._build_backbone().to(self.device)
            projector = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128).to(self.device)

            params = list(backbone.parameters()) + list(projector.parameters())
            optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

            # Training
            losses: List[float] = []
            best_loss = float("inf")
            best_ckpt_path = os.path.join(self.config.checkpoints_dir, "best_backbone.pt")

            for epoch in range(1, self.config.epochs + 1):
                backbone.train()
                projector.train()

                running_loss = 0.0
                num_batches = 0

                for x1, x2 in loader:
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)

                    h1 = backbone(x1)  # [B, 512]
                    h2 = backbone(x2)  # [B, 512]
                    z1 = projector(h1) # [B, 128]
                    z2 = projector(h2) # [B, 128]

                    loss = nt_xent_loss(z1, z2, temperature=self.config.temperature)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1

                epoch_loss = running_loss / max(num_batches, 1)
                losses.append(epoch_loss)
                logging.info(f"[SSL] Epoch {epoch}/{self.config.epochs} | loss={epoch_loss:.4f}")

                # save best checkpoint (by loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self._save_checkpoint(best_ckpt_path, backbone, projector, epoch)

            # Write metrics JSON
            metrics_payload = {
                "device": str(self.device),
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "temperature": self.config.temperature,
                "loss_curve": losses,
                "final_loss": losses[-1] if losses else None,
                "best_loss": best_loss,
                "best_backbone_path": best_ckpt_path,
            }
            self._write_json(self.config.ssl_metrics_file_path, metrics_payload)


            # log final + best
            log_metrics({"final_loss": losses[-1], "best_loss": best_loss})
            log_artifact(self.config.ssl_metrics_file_path)
            log_artifact(best_ckpt_path)
            end_run()

            logging.info("[SSL] Representation learning completed successfully.")

            return RepresentationLearningArtifact(
                ssl_dir=self.config.ssl_dir,
                checkpoints_dir=self.config.checkpoints_dir,
                metrics_dir=self.config.metrics_dir,
                best_backbone_path=best_ckpt_path,
                ssl_metrics_file_path=self.config.ssl_metrics_file_path,
                total_epochs=self.config.epochs,
                final_loss=float(losses[-1]) if losses else float("inf"),
            )

        except Exception as e:
            raise CustomException(e, sys)
