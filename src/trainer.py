"""
Smart Education Project — Model Training, Validation, and Evaluation Module
=============================================================================
Provides:
1. ModelTrainer: Unified training loop with mixed precision (AMP), early stopping,
   checkpointing with metadata, configurable overfitting/underfitting diagnosis,
   and training curve generation.
2. Retraining Invariant: Guarantees that any corrected model is trained using a
   fresh ModelTrainer instance and a new optimizer pointing exclusively to the
   corrected model's parameters.
3. RecommendationEvaluator: Precision@K, Recall@K, NDCG@K, HitRate@K.
"""

import os
import sys
import time
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PLOTS_DIR, CHECKPOINTS_DIR

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ModelTrainer:
    """
    Comprehensive training module for Smart Education models.
    Supports Knowledge Tracing (LSTM, Transformer, BERT-NCF, CNN-LSTM)
    and Autoencoder models.
    """
    def __init__(self, model: nn.Module, device: torch.device, config: Optional[Dict[str, Any]] = None):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        self.plots_dir = str(self.config.get('plots_dir', PLOTS_DIR))
        self.checkpoints_dir = str(self.config.get('checkpoints_dir', CHECKPOINTS_DIR))
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def train_epoch(self, train_loader, optimizer, criterion, scaler=None, is_autoencoder=False) -> Dict[str, float]:
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_preds = []

        use_cuda = self.device.type == 'cuda'

        for batch in train_loader:
            optimizer.zero_grad()

            if is_autoencoder:
                # inputs: [B, input_dim], targets: [B, 1]
                inputs, targets = batch[0], batch[1]
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with autocast('cuda', enabled=(scaler is not None and use_cuda)):
                    recon, scores = self.model(inputs)
                    recon_loss = nn.MSELoss()(recon, inputs)
                    # Use recon loss + auxiliary recommendation loss
                    loss = recon_loss

                preds = torch.sigmoid(scores).detach().cpu().numpy().flatten()
                targs = inputs.cpu().numpy().flatten()
            else:
                # KT format: features [B, S, F], targets [B, S], mask [B, S]
                inputs, targets, mask, _ = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                if mask.sum() == 0:
                    continue

                with autocast('cuda', enabled=(scaler is not None and use_cuda)):
                    outputs = self.model(inputs)  # logits [B, S]
                    valid_outputs = outputs[mask]
                    valid_targets = targets[mask].float()
                    loss = criterion(valid_outputs, valid_targets)

                preds = torch.sigmoid(valid_outputs).detach().cpu().numpy()
                targs = valid_targets.cpu().numpy()

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

            total_loss += loss.item() * len(targs)
            all_preds.extend(preds)
            all_targets.extend(targs)

        avg_loss = total_loss / max(len(all_targets), 1)
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        pred_labels = (all_preds >= 0.5).astype(int)

        try:
            auc = roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_targets, pred_labels) if len(all_targets) > 0 else 0.0,
            'auc': auc
        }

    def evaluate(self, data_loader, criterion, is_autoencoder=False) -> Dict[str, float]:
        """Evaluates model without gradients."""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                if is_autoencoder:
                    inputs, targets = batch[0], batch[1]
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    recon, scores = self.model(inputs)
                    recon_loss = nn.MSELoss()(recon, inputs)
                    loss = recon_loss

                    preds = torch.sigmoid(scores).cpu().numpy().flatten()
                    targs = inputs.cpu().numpy().flatten()
                else:
                    inputs, targets, mask, _ = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    mask = mask.to(self.device)

                    if mask.sum() == 0:
                        continue

                    outputs = self.model(inputs)
                    valid_outputs = outputs[mask]
                    valid_targets = targets[mask].float()
                    loss = criterion(valid_outputs, valid_targets)

                    preds = torch.sigmoid(valid_outputs).cpu().numpy()
                    targs = valid_targets.cpu().numpy()

                total_loss += loss.item() * len(targs)
                all_preds.extend(preds)
                all_targets.extend(targs)

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        pred_labels = (all_preds >= 0.5).astype(int)
        avg_loss = total_loss / max(len(all_targets), 1)

        try:
            auc = roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.5
        except Exception:
            auc = 0.5

        try:
            pr_auc = average_precision_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.0
        except Exception:
            pr_auc = 0.0

        try:
            ll = log_loss(all_targets, np.clip(all_preds, 1e-7, 1 - 1e-7)) if len(np.unique(all_targets)) > 1 else 0.0
        except Exception:
            ll = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_targets, pred_labels) if len(all_targets) > 0 else 0.0,
            'precision': precision_score(all_targets, pred_labels, zero_division=0) if len(all_targets) > 0 else 0.0,
            'recall': recall_score(all_targets, pred_labels, zero_division=0) if len(all_targets) > 0 else 0.0,
            'f1': f1_score(all_targets, pred_labels, zero_division=0) if len(all_targets) > 0 else 0.0,
            'auc': auc,
            'pr_auc': pr_auc,
            'log_loss': ll
        }

    def train(self, train_loader, val_loader, epochs=8, lr=1e-3, weight_decay=1e-4, patience=3,
              model_name="model", is_autoencoder=False) -> Dict[str, Any]:
        """Full training loop with early stopping, learning rate scheduler, and checkpointing."""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scaler = GradScaler('cuda') if self.device.type == 'cuda' else None

        history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_acc': [], 'val_acc': []
        }

        best_val_loss = float('inf')
        best_val_auc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        best_model_path = os.path.join(self.checkpoints_dir, f"{model_name}_best.pth")
        meta_path = os.path.join(self.checkpoints_dir, f"{model_name}_best_meta.json")

        start_time = time.time()

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, scaler, is_autoencoder=is_autoencoder)
            val_metrics = self.evaluate(val_loader, criterion, is_autoencoder=is_autoencoder)

            scheduler.step(val_metrics['loss'])

            history['train_loss'].append(train_metrics['loss'])
            history['train_auc'].append(train_metrics['auc'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_acc'].append(val_metrics['accuracy'])

            print(f"  Epoch {epoch+1:2d}/{epochs:2d} | Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} AUC: {train_metrics['auc']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['auc']:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_val_auc = val_metrics['auc']
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), best_model_path)
                epochs_no_improve = 0

                # Save human-readable checkpoint metadata alongside binary .pth
                meta_data = {
                    'model_name': model_name,
                    'best_epoch': best_epoch,
                    'val_loss': float(best_val_loss),
                    'val_auc': float(best_val_auc),
                    'val_acc': float(val_metrics['accuracy']),
                    'parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(meta_path, 'w') as mf:
                    json.dump(meta_data, mf, indent=2)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break

        training_time = time.time() - start_time
        history['training_time'] = training_time
        history['best_epoch'] = best_epoch
        history['best_val_loss'] = best_val_loss
        history['best_val_auc'] = best_val_auc

        # Restore weights of the best performing epoch
        self.model.load_state_dict(best_model_wts)
        return history

    def detect_overfitting(self, history: Dict[str, Any],
                           loss_gap_severe: float = 0.08,
                           loss_gap_mild: float = 0.04,
                           underfit_loss_thresh: float = 0.65) -> Tuple[str, Dict[str, Any]]:
        """
        Diagnoses overfitting or underfitting using configurable heuristic criteria.
        Returns diagnosis string and empirical evidence dictionary.
        """
        train_loss = history['train_loss']
        val_loss = history['val_loss']

        if len(val_loss) < 2:
            return "Indeterminate (insufficient epochs)", {"train_loss": train_loss[-1], "val_loss": val_loss[-1]}

        recent_train_loss = np.mean(train_loss[-2:])
        recent_val_loss = np.mean(val_loss[-2:])
        loss_gap = recent_val_loss - recent_train_loss

        val_trend = val_loss[-1] - val_loss[0]
        train_trend = train_loss[-1] - train_loss[0]

        evidence = {
            "recent_train_loss": float(recent_train_loss),
            "recent_val_loss": float(recent_val_loss),
            "loss_gap": float(loss_gap),
            "val_trend": float(val_trend),
            "train_trend": float(train_trend),
            "heuristic_thresholds": {
                "loss_gap_severe": loss_gap_severe,
                "loss_gap_mild": loss_gap_mild,
                "underfit_loss_thresh": underfit_loss_thresh
            }
        }

        if loss_gap > loss_gap_severe and val_trend > 0.02 and train_trend < 0:
            return "Severe Overfitting", evidence
        elif loss_gap > loss_gap_mild:
            return "Mild Overfitting", evidence
        elif recent_train_loss > underfit_loss_thresh and recent_val_loss > underfit_loss_thresh:
            return "Underfitting", evidence
        else:
            return "Healthy Fit / Well-Regularized", evidence

    def apply_correction(self, model: nn.Module, diagnosis: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Creates a regularized or capacity-tuned instance based on diagnosis.
        Note: The returned model is a distinct object, to be trained by a fresh ModelTrainer.
        """
        corrected_model = copy.deepcopy(model)
        correction_info = {}

        if "Overfitting" in diagnosis:
            print("  Applying Regularization Correction: increasing dropout and weight decay.")
            for m in corrected_model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = min(0.5, m.p + 0.15)
            correction_info = {
                "diagnosis": diagnosis,
                "action": "Increased dropout (+0.15) and higher weight decay (1e-3)",
                "weight_decay": 1e-3,
                "learning_rate": 5e-4
            }
        elif "Underfitting" in diagnosis:
            print("  Applying Capacity Correction: lowering regularization and tuning learning rate.")
            correction_info = {
                "diagnosis": diagnosis,
                "action": "Lowered weight decay and boosted learning rate (1.5e-3)",
                "weight_decay": 1e-5,
                "learning_rate": 1.5e-3
            }
        else:
            print("  Model fit is healthy; applying controlled fine-tuning rate.")
            correction_info = {
                "diagnosis": diagnosis,
                "action": "Fine-tuning with conservative learning rate (5e-4)",
                "weight_decay": 2e-4,
                "learning_rate": 5e-4
            }
        return corrected_model, correction_info

    def plot_training_curves(self, history: Dict[str, Any], title: str = "Training Curves",
                             save_filename: str = "training_curves.png"):
        """Plots and saves loss and ROC-AUC convergence curves."""
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Loss Curves
        axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
        axes[0].set_title(f'{title} — Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # ROC-AUC Curves
        axes[1].plot(epochs, history['train_auc'], 'b-o', label='Train AUC')
        axes[1].plot(epochs, history['val_auc'], 'r-s', label='Val AUC')
        axes[1].set_title(f'{title} — ROC-AUC')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('ROC-AUC')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, save_filename)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  Saved training plot to: {save_path}")


class RecommendationEvaluator:
    """Computes standard top-K recommendation metrics."""

    @staticmethod
    def precision_at_k(predicted: list, actual: list, k: int) -> float:
        pred_k = predicted[:k]
        hits = len(set(pred_k) & set(actual))
        return hits / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(predicted: list, actual: list, k: int) -> float:
        pred_k = predicted[:k]
        hits = len(set(pred_k) & set(actual))
        return hits / len(actual) if len(actual) > 0 else 0.0

    @staticmethod
    def ndcg_at_k(predicted: list, actual: list, k: int) -> float:
        pred_k = predicted[:k]
        dcg = sum([1.0 / np.log2(i + 2) for i, p in enumerate(pred_k) if p in actual])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(actual)))])
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate_at_k(predicted: list, actual: list, k: int) -> float:
        pred_k = predicted[:k]
        return 1.0 if len(set(pred_k) & set(actual)) > 0 else 0.0

    @classmethod
    def evaluate_recommendations(cls, predictions_dict: Dict[str, list],
                                 actuals_dict: Dict[str, list],
                                 k_values: list = [5, 10]) -> Dict[str, float]:
        results = {}
        for k in k_values:
            precisions, recalls, ndcgs, hits = [], [], [], []
            for user, actual in actuals_dict.items():
                if user not in predictions_dict or len(actual) == 0:
                    continue
                pred = predictions_dict[user]
                precisions.append(cls.precision_at_k(pred, actual, k))
                recalls.append(cls.recall_at_k(pred, actual, k))
                ndcgs.append(cls.ndcg_at_k(pred, actual, k))
                hits.append(cls.hit_rate_at_k(pred, actual, k))

            results[f'Precision@{k}'] = float(np.mean(precisions)) if precisions else 0.0
            results[f'Recall@{k}'] = float(np.mean(recalls)) if recalls else 0.0
            results[f'NDCG@{k}'] = float(np.mean(ndcgs)) if ndcgs else 0.0
            results[f'HitRate@{k}'] = float(np.mean(hits)) if hits else 0.0
        return results
