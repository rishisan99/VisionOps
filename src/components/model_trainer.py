# src/components/model_trainer.py

import os
import sys
import json
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import FineTuneModelConfig
from src.entity.artifact_entity import (
    DataSplittingArtifact,
    RepresentationLearningArtifact,
    ModelTrainerArtifact,
)
from src.components.data_loader import create_supervised_dataloader, get_device

import mlflow
from src.utils.mlflow_utils import (
    setup_mlflow,
    start_run,
    log_params,
    log_metrics,
    log_artifact,
    end_run
)


from src.entity.config_entity import MLflowConfig


def compute_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1_from_cm(cm: torch.Tensor) -> Tuple[float, float, float]:
    """
    Macro precision/recall/F1 from confusion matrix.
    """
    num_classes = cm.size(0)
    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return float(sum(precisions) / num_classes), float(sum(recalls) / num_classes), float(sum(f1s) / num_classes)


class FineTuneTrainer:
    def __init__(
        self,
        finetune_config: FineTuneModelConfig,
        data_splitting_artifact: DataSplittingArtifact,
        ssl_artifact: RepresentationLearningArtifact,
    ):
        self.config = finetune_config
        self.data_artifact = data_splitting_artifact
        self.ssl_artifact = ssl_artifact
        self.device = get_device()

    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _save_confusion_matrix_png(self, cm: torch.Tensor, idx_to_class: Dict[int, str], out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig = plt.figure()
        plt.imshow(cm.numpy(), interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        labels = [idx_to_class[i] for i in range(len(idx_to_class))]
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)

        # annotate
        for i in range(cm.size(0)):
            for j in range(cm.size(1)):
                plt.text(j, i, int(cm[i, j].item()), ha="center", va="center")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    def _build_model(self, num_classes: int) -> nn.Module:
        """
        ResNet18 backbone + linear classifier.
        """
        model = models.resnet18(weights=None)
        model.fc = nn.Identity()

        classifier = nn.Linear(512, num_classes)

        # Wrap so forward returns logits
        class Model(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                feats = self.backbone(x)
                logits = self.head(feats)
                return logits

        return Model(model, classifier)

    def _load_ssl_backbone(self, model: nn.Module) -> None:
        """
        Load backbone weights from best_backbone.pt saved in SSL stage.
        """
        ckpt = torch.load(self.ssl_artifact.best_backbone_path, map_location="cpu")
        backbone_sd = ckpt["backbone_state_dict"]

        missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
        logging.info(f"[FineTune] Loaded SSL backbone. missing={len(missing)} unexpected={len(unexpected)}")

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader, num_classes: int) -> Tuple[float, torch.Tensor]:
        model.eval()
        correct = 0
        total = 0
        y_true, y_pred = [], []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        acc = correct / max(total, 1)
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
        return float(acc), cm

    def _set_trainable(self, model: nn.Module, freeze_backbone: bool, unfreeze_layer4: bool = False) -> None:
        # Freeze everything in backbone
        for p in model.backbone.parameters():
            p.requires_grad = not freeze_backbone

        # If backbone is frozen but we want to unfreeze only layer4
        if freeze_backbone and unfreeze_layer4:
            for p in model.backbone.layer4.parameters():
                p.requires_grad = True

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"[FineTune] Using device: {self.device}")

            # Dataloaders
            train_loader, class_to_idx = create_supervised_dataloader(
                data_dir=self.data_artifact.train_dir,
                batch_size=self.config.batch_size,
                train=True,
                num_workers=0  # ✅ safest on macOS
            )
            val_loader, _ = create_supervised_dataloader(
                data_dir=self.data_artifact.val_dir,
                batch_size=self.config.batch_size,
                train=False,
                num_workers=0
            )
            test_loader, _ = create_supervised_dataloader(
                data_dir=self.data_artifact.test_dir,
                batch_size=self.config.batch_size,
                train=False,
                num_workers=0
            )

            num_classes = len(class_to_idx)
            idx_to_class = {v: k for k, v in class_to_idx.items()}

            # Model
            model = self._build_model(num_classes=num_classes)
            self._load_ssl_backbone(model)

            # -------------------------
            # Phase 1: head-only training
            # -------------------------
            self._set_trainable(model, freeze_backbone=True, unfreeze_layer4=False)
            logging.info("[FineTune] Phase 1: Backbone frozen. Training classifier head only.")

            model = model.to(self.device)

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.config.learning_rate
            )

            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0.0
            best_model_path = os.path.join(self.config.checkpoints_dir, "classifier.pt")

            criterion = nn.CrossEntropyLoss()

            mlflow_cfg = MLflowConfig()
            setup_mlflow(mlflow_cfg.experiment_name, mlflow_cfg.tracking_uri)

            run = start_run(
                run_name="finetune_resnet18_classifier",
                tags={"stage": "finetune", "model": "resnet18"}
            )

            log_params({
                "batch_size": self.config.batch_size,
                "phase1_epochs": self.config.phase1_epochs,
                "phase2_epochs": self.config.phase2_epochs,
                "lr_phase1": self.config.learning_rate,
                "lr_phase2": self.config.phase2_learning_rate,
                "freeze_backbone": True,
                "device": str(self.device),
            })
            
            # -------------------------
            # Phase 1 optimizer (head only)
            # -------------------------
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.config.learning_rate
            )

            # -------------------------
            # PHASE 1: head-only epochs
            # -------------------------
            for epoch in range(1, self.config.phase1_epochs + 1):
                model.train()
                running_loss = 0.0
                batches = 0

                for x, y in train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = model(x)
                    loss = criterion(logits, y)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    batches += 1

                train_loss = running_loss / max(batches, 1)
                val_acc, _ = self._evaluate(model, val_loader, num_classes=num_classes)

                logging.info(f"[FineTune] Phase1 Epoch {epoch}/{self.config.phase1_epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(self.config.checkpoints_dir, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "class_to_idx": class_to_idx,
                            "epoch": epoch,
                            "val_acc": val_acc,
                            "phase": "phase1",
                        },
                        best_model_path
                    )

            # -------------------------
            # PHASE 2: unfreeze layer4 (optional)
            # -------------------------
            if self.config.phase2_unfreeze_layer4 and self.config.phase2_epochs > 0:
                self._set_trainable(model, freeze_backbone=True, unfreeze_layer4=True)
                logging.info("[FineTune] Phase 2: Unfroze backbone.layer4 + head (fine-tuning last block).")

                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=self.config.phase2_learning_rate
                )

                for epoch in range(1, self.config.phase2_epochs + 1):
                    model.train()
                    running_loss = 0.0
                    batches = 0

                    for x, y in train_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        logits = model(x)
                        loss = criterion(logits, y)

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        batches += 1

                    train_loss = running_loss / max(batches, 1)
                    val_acc, _ = self._evaluate(model, val_loader, num_classes=num_classes)

                    logging.info(f"[FineTune] Phase2 Epoch {epoch}/{self.config.phase2_epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "class_to_idx": class_to_idx,
                                "epoch": epoch,
                                "val_acc": val_acc,
                                "phase": "phase2_layer4",
                            },
                            best_model_path
                        )

            # Load best model for final evaluation
            best_ckpt = torch.load(best_model_path, map_location="cpu")
            model.load_state_dict(best_ckpt["model_state_dict"])
            model = model.to(self.device)

            test_acc, cm = self._evaluate(model, test_loader, num_classes=num_classes)
            precision, recall, f1 = precision_recall_f1_from_cm(cm)

            # Save confusion matrix image
            self._save_confusion_matrix_png(cm, idx_to_class, self.config.confusion_matrix_path)

            # Save metrics json
            metrics_payload = {
                "device": str(self.device),
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "freeze_backbone": self.config.freeze_backbone,
                "best_val_acc": best_val_acc,
                "test_accuracy": test_acc,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "model_checkpoint": best_model_path,
            }
            self._write_json(self.config.metrics_file_path, metrics_payload)

            log_metrics({
                "test_accuracy": test_acc,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            })

            # Log artifacts
            log_artifact(self.config.metrics_file_path)
            log_artifact(self.config.confusion_matrix_path)
            log_artifact(best_model_path)

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/classifier.pt"

            mlflow.register_model(
                model_uri=model_uri,
                name="tomato_leaf_classifier"
            )

            end_run()

            logging.info("[FineTune] Supervised fine-tuning completed successfully.")

            return ModelTrainerArtifact(
                finetune_dir=self.config.finetune_dir,
                checkpoints_dir=self.config.checkpoints_dir,
                metrics_dir=self.config.metrics_dir,
                trained_model_path=best_model_path,
                metrics_file_path=self.config.metrics_file_path,
                confusion_matrix_path=self.config.confusion_matrix_path,
                accuracy=float(test_acc),
                f1_score=float(f1),
                precision=float(precision),
                recall=float(recall),
            )

        except Exception as e:
            raise CustomException(e, sys)
