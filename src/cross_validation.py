"""
Smart Education Project — 10-Fold Grouped Cross-Validation Module
===================================================================
Executes 10-Fold Grouped Cross-Validation grouped strictly by actual student `user_id`.
Ensures:
1. Zero student overlap between training and validation partitions in every fold (Dev_train ∩ Dev_val = ∅).
2. Per-fold logging of:
   - Fold number
   - Training user count
   - Validation user count
   - Training event count
   - Validation event count
   - Training/Validation user overlap (strictly 0)
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss
3. Generates reports/cv_report.md and reports/model_cv_10fold.csv.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, log_loss
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR, RANDOM_SEED, DEFAULT_BATCH_SIZE, ensure_dirs
from src.dataset import KTSequenceDataset
from src.trainer import ModelTrainer

ensure_dirs()
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class GroupedCrossValidator:
    """
    Executes 10-Fold GroupKFold Cross-Validation strictly grouped by actual student user_id.
    """
    def __init__(self, model_class, model_kwargs: Dict[str, Any], device: torch.device):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.device = device

    def run_grouped_kfold(self, sequence_dataset: KTSequenceDataset,
                          n_splits: int = 10, epochs: int = 2,
                          lr: float = 1e-3, batch_size: int = DEFAULT_BATCH_SIZE,
                          model_name: str = "Model") -> Dict[str, Any]:
        """
        Executes grouped cross-validation on sequence dataset using student user IDs.
        """
        print("\n" + "=" * 80)
        print(f"  10-FOLD GROUPED CROSS-VALIDATION ({model_name.upper()})")
        print("  Grouping Strategy: GroupKFold by actual student user_id (Zero Overlap Enforced)")
        print("=" * 80)

        all_sequences = sequence_dataset.sequences
        user_ids = sequence_dataset.user_ids

        assert len(all_sequences) == len(user_ids), "Sequences and user_ids length mismatch!"

        gkf = GroupKFold(n_splits=n_splits)
        fold_records = []

        print(f"  Total Students: {len(user_ids):,} | Splits: {n_splits}")
        print(f"\n  {'Fold':>4} | {'Train Users':>11} {'Val Users':>10} {'Overlap':>8} | {'Train Evts':>10} {'Val Evts':>10} | {'Acc':>7} {'F1':>7} {'AUC':>7} {'PR-AUC':>7}")
        print(f"  {'-'*95}")

        for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(all_sequences, groups=user_ids)):
            train_uids = set([user_ids[i] for i in train_indices])
            val_uids = set([user_ids[i] for i in val_indices])
            overlap = len(train_uids.intersection(val_uids))

            assert overlap == 0, f"FATAL ERROR: Student overlap detected in Fold {fold_idx + 1}!"

            train_events = sum(len(all_sequences[i]) for i in train_indices)
            val_events = sum(len(all_sequences[i]) for i in val_indices)

            train_sub = Subset(sequence_dataset, train_indices)
            val_sub = Subset(sequence_dataset, val_indices)

            train_loader = DataLoader(
                train_sub, batch_size=batch_size, shuffle=True,
                collate_fn=KTSequenceDataset.collate_fn
            )
            val_loader = DataLoader(
                val_sub, batch_size=batch_size, shuffle=False,
                collate_fn=KTSequenceDataset.collate_fn
            )

            # Instantiate a fresh model and trainer for every single fold
            fold_model = self.model_class(**self.model_kwargs).to(self.device)
            trainer = ModelTrainer(fold_model, self.device)
            optimizer = optim.AdamW(fold_model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(epochs):
                trainer.train_epoch(train_loader, optimizer, criterion)

            val_metrics = trainer.evaluate(val_loader, criterion)

            record = {
                'fold': fold_idx + 1,
                'train_users': len(train_uids),
                'val_users': len(val_uids),
                'user_overlap': overlap,
                'train_events': train_events,
                'val_events': val_events,
                'accuracy': val_metrics['accuracy'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1': val_metrics['f1'],
                'roc_auc': val_metrics['auc'],
                'pr_auc': val_metrics['pr_auc'],
                'log_loss': val_metrics['log_loss']
            }
            fold_records.append(record)

            print(f"  {fold_idx+1:>4d} | {len(train_uids):>11,d} {len(val_uids):>10,d} {overlap:>8d} | {train_events:>10,d} {val_events:>10,d} | "
                  f"{val_metrics['accuracy']:>7.4f} {val_metrics['f1']:>7.4f} {val_metrics['auc']:>7.4f} {val_metrics['pr_auc']:>7.4f}")

        df_results = pd.DataFrame(fold_records)
        means = df_results.mean(numeric_only=True)
        stds = df_results.std(numeric_only=True)

        print(f"  {'-'*95}")
        print(f"  MEAN | {int(means['train_users']):>11,d} {int(means['val_users']):>10,d} {int(means['user_overlap']):>8d} | "
              f"{int(means['train_events']):>10,d} {int(means['val_events']):>10,d} | "
              f"{means['accuracy']:>7.4f} {means['f1']:>7.4f} {means['roc_auc']:>7.4f} {means['pr_auc']:>7.4f}")
        print(f"  STD  | {'±0':>11} {'±0':>10} {'0':>8} | "
              f"±{int(stds['train_events']):>9,d} ±{int(stds['val_events']):>9,d} | "
              f"±{stds['accuracy']:>6.4f} ±{stds['f1']:>6.4f} ±{stds['roc_auc']:>6.4f} ±{stds['pr_auc']:>6.4f}")

        # Save results CSV
        csv_filename = f"{model_name.lower().replace(' ', '_')}_cv_10fold.csv"
        df_results.to_csv(REPORTS_DIR / csv_filename, index=False)
        df_results.to_csv(REPORTS_DIR / "model_cv_10fold.csv", index=False)

        # Write markdown report
        self._write_cv_report(df_results, means, stds, model_name)

        summary = {
            'accuracy_mean': float(means['accuracy']), 'accuracy_std': float(stds['accuracy']),
            'precision_mean': float(means['precision']), 'precision_std': float(stds['precision']),
            'recall_mean': float(means['recall']), 'recall_std': float(stds['recall']),
            'f1_mean': float(means['f1']), 'f1_std': float(stds['f1']),
            'roc_auc_mean': float(means['roc_auc']), 'roc_auc_std': float(stds['roc_auc']),
            'pr_auc_mean': float(means['pr_auc']), 'pr_auc_std': float(stds['pr_auc']),
            'log_loss_mean': float(means['log_loss']), 'log_loss_std': float(stds['log_loss'])
        }
        return {'fold_records': fold_records, 'summary': summary, 'dataframe': df_results}

    def _write_cv_report(self, df_results: pd.DataFrame, means: pd.Series, stds: pd.Series, model_name: str):
        report_path = REPORTS_DIR / "cv_report.md"
        lines = []
        lines.append("# 10-Fold Grouped Cross-Validation Report (Development Cohort)")
        lines.append(f"**Evaluated Model:** {model_name}  ")
        lines.append("**Grouping Key:** Actual student `user_id` (Non-overlapping GroupKFold)  ")
        lines.append(f"**Total Folds:** 10  ")
        lines.append(f"**Cross-Cohort User Overlap:** 0 (PASSED: Strict student-level partition isolation)\n")

        lines.append("## Per-Fold Empirical Metrics & Cohort Statistics")
        lines.append("| Fold | Train Users | Val Users | Overlap | Train Events | Val Events | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Log Loss |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

        for _, r in df_results.iterrows():
            lines.append(
                f"| Fold {int(r['fold'])} | {int(r['train_users']):,} | {int(r['val_users']):,} | {int(r['user_overlap'])} | "
                f"{int(r['train_events']):,} | {int(r['val_events']):,} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['log_loss']:.4f} |"
            )

        lines.append(
            f"| **Mean** | **{int(means['train_users']):,}** | **{int(means['val_users']):,}** | **0** | "
            f"**{int(means['train_events']):,}** | **{int(means['val_events']):,}** | "
            f"**{means['accuracy']:.4f}** | **{means['precision']:.4f}** | **{means['recall']:.4f}** | "
            f"**{means['f1']:.4f}** | **{means['roc_auc']:.4f}** | **{means['pr_auc']:.4f}** | **{means['log_loss']:.4f}** |"
        )
        lines.append(
            f"| **Std** | ±0 | ±0 | 0 | ±{int(stds['train_events']):,} | ±{int(stds['val_events']):,} | "
            f"±{stds['accuracy']:.4f} | ±{stds['precision']:.4f} | ±{stds['recall']:.4f} | "
            f"±{stds['f1']:.4f} | ±{stds['roc_auc']:.4f} | ±{stds['pr_auc']:.4f} | ±{stds['log_loss']:.4f} |"
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Saved CV Report to: {report_path.name}")
