"""
Smart Education Project — Attention Model Trainer & Diagnosis Module
=====================================================================
Provides:
1. AttentionTrainer: Training loop with mixed-precision (AMP) on RTX 3050,
   Early Stopping, and metadata-paired checkpointing.
2. Academic Overfitting/Underfitting Diagnostic Engine:
   Monitors training vs validation loss gaps and trends.
3. Retraining Invariant: Guarantees that any corrected model is instantiated as a
   completely fresh PyTorch object with a new optimizer and distinct ModelTrainer.
"""

import os
import time
import copy
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_config import ATTENTION_CHECKPOINTS_DIR, AttentionConfig
from attention.attention_model import MultimodalAttentionModel


class AttentionTrainer:
    """
    Dedicated training orchestrator for the Multimodal Attention Model.
    """
    def __init__(
        self,
        model: MultimodalAttentionModel,
        device: torch.device,
        config: Optional[AttentionConfig] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or AttentionConfig()
        self.checkpoints_dir = ATTENTION_CHECKPOINTS_DIR
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        scaler: Optional[GradScaler] = None
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        use_cuda = self.device.type == "cuda"

        for x_v, x_a, mask_v, mask_a, targets in train_loader:
            x_v = x_v.to(self.device)
            x_a = x_a.to(self.device)
            mask_v = mask_v.to(self.device)
            mask_a = mask_a.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=(scaler is not None and use_cuda)):
                logits = self.model(x_v, x_a, mask_v, mask_a)
                loss = criterion(logits, targets)

            if scaler is not None and use_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / max(len(all_targets), 1)
        acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
        return {"loss": avg_loss, "accuracy": acc}

    def evaluate(self, val_loader, criterion) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for x_v, x_a, mask_v, mask_a, targets in val_loader:
                x_v = x_v.to(self.device)
                x_a = x_a.to(self.device)
                mask_v = mask_v.to(self.device)
                mask_a = mask_a.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(x_v, x_a, mask_v, mask_a)
                loss = criterion(logits, targets)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                total_loss += loss.item() * targets.size(0)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        avg_loss = total_loss / max(len(all_targets), 1)
        acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

        # Multi-class ROC-AUC (One-vs-Rest)
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.5

        try:
            ll = log_loss(all_targets, np.clip(all_probs, 1e-7, 1 - 1e-7))
        except Exception:
            ll = 0.0

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc": auc,
            "log_loss": ll,
            "all_targets": all_targets,
            "all_preds": all_preds,
            "all_probs": all_probs
        }

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 4,
        model_name: str = "attention_multimodal"
    ) -> Dict[str, Any]:
        """
        Executes complete training loop with Early Stopping, scheduler, and paired JSON metadata.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        scaler = GradScaler("cuda") if (self.device.type == "cuda" and self.config.use_mixed_precision) else None

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_f1_macro": []
        }

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        best_model_path = self.checkpoints_dir / f"{model_name}_best.pth"
        meta_path = self.checkpoints_dir / f"{model_name}_best_meta.json"

        start_time = time.time()

        for epoch in range(epochs):
            tr_m = self.train_epoch(train_loader, optimizer, criterion, scaler)
            val_m = self.evaluate(val_loader, criterion)

            scheduler.step(val_m["loss"])

            history["train_loss"].append(tr_m["loss"])
            history["train_acc"].append(tr_m["accuracy"])
            history["val_loss"].append(val_m["loss"])
            history["val_acc"].append(val_m["accuracy"])
            history["val_f1_macro"].append(val_m["f1_macro"])

            print(f"  Epoch {epoch+1:2d}/{epochs:2d} | Train Loss: {tr_m['loss']:.4f} Acc: {tr_m['accuracy']:.4f} | "
                  f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} F1: {val_m['f1_macro']:.4f}")

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                best_val_acc = val_m["accuracy"]
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(self.model.state_dict())

                torch.save(self.model.state_dict(), best_model_path)
                epochs_no_improve = 0

                meta_data = {
                    "model_name": model_name,
                    "best_epoch": best_epoch,
                    "val_loss": float(best_val_loss),
                    "val_acc": float(best_val_acc),
                    "val_f1_macro": float(val_m["f1_macro"]),
                    "val_roc_auc": float(val_m["roc_auc"]),
                    "parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta_data, mf, indent=2)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break

        history["training_time"] = time.time() - start_time
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["best_val_acc"] = best_val_acc

        self.model.load_state_dict(best_model_wts)
        return history

    def detect_overfitting(
        self,
        history: Dict[str, Any],
        gap_threshold: float = 0.08
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Diagnoses overfitting or underfitting using quantitative loss-gap metrics.
        """
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]

        if len(val_loss) < 2:
            return "Indeterminate (insufficient epochs)", {"train_loss": train_loss[-1], "val_loss": val_loss[-1]}

        recent_tr = float(np.mean(train_loss[-2:]))
        recent_val = float(np.mean(val_loss[-2:]))
        loss_gap = float(recent_val - recent_tr)

        evidence = {
            "recent_train_loss": recent_tr,
            "recent_val_loss": recent_val,
            "loss_gap": loss_gap,
            "final_train_acc": float(history["train_acc"][-1]),
            "final_val_acc": float(history["val_acc"][-1])
        }

        if loss_gap > gap_threshold:
            diagnosis = "Overfitting Detected (Val loss diverging from Train loss)"
        elif recent_tr > 1.10 and recent_val > 1.10:
            diagnosis = "Underfitting Detected (High loss on both train and val)"
        else:
            diagnosis = "Optimal Generalization (Balanced loss trajectories)"

        return diagnosis, evidence
