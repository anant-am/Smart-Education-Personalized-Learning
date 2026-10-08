"""
Smart Education Project — Training-Only SMOTE Utility Module
=============================================================
Academic Defense Compliant SMOTE Module:
1. Applies SMOTE exclusively to the Training partition (`train.csv`) of the Development cohort.
2. Preserves Validation (`val.csv`) and Unseen Test (`test.csv`) sets 100% UNTOUCHED in their natural class distribution.
3. Operates on derived multi-feature tabular learner-state representations, avoiding corruption of chronological sequence graphs.
4. Logs exact class distributions before and after SMOTE and outputs `reports/smote_report.md`.
"""

import os
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, ensure_dirs

ensure_dirs()
np.random.seed(RANDOM_SEED)


class AcademicSMOTEHandler:
    """
    Handles academically compliant oversampling on training data only.
    """
    def __init__(self, random_seed: int = RANDOM_SEED, k_neighbors: int = 5):
        self.random_seed = random_seed
        self.k_neighbors = k_neighbors
        self.smote = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)

    def apply_smote_to_training(self, train_df: pd.DataFrame,
                                feature_cols: list = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fits and resamples the training set tabular representation.
        """
        if feature_cols is None:
            feature_cols = [
                'previous_accuracy', 'recent_accuracy_5', 'attempt_count',
                'response_time_norm', 'time_since_prev_norm',
                'source_encoded', 'platform_encoded', 'part', 'num_responses',
                'question_idx'
            ]

        # Verify required columns exist
        available_cols = [c for c in feature_cols if c in train_df.columns]
        train_clean = train_df.dropna(subset=available_cols + ['is_correct']).copy()

        X_tr = train_clean[available_cols].values
        y_tr = train_clean['is_correct'].values.astype(int)

        before_counts = Counter(y_tr)
        print("\n" + "=" * 75)
        print("  ACADEMIC SMOTE APPLICATION (TRAINING DATA ONLY)")
        print("=" * 75)
        print(f"  Class 0 (Incorrect): {before_counts[0]:>8,d} ({before_counts[0]/len(y_tr)*100:.2f}%)")
        print(f"  Class 1 (Correct):   {before_counts[1]:>8,d} ({before_counts[1]/len(y_tr)*100:.2f}%)")
        print(f"  Initial Imbalance Ratio (0:1): {before_counts[0]/before_counts[1]:.3f}")

        # Resample
        X_resampled, y_resampled = self.smote.fit_resample(X_tr, y_tr)
        after_counts = Counter(y_resampled)

        print(f"\n  Resampled Class 0:   {after_counts[0]:>8,d} ({after_counts[0]/len(y_resampled)*100:.2f}%)")
        print(f"  Resampled Class 1:   {after_counts[1]:>8,d} ({after_counts[1]/len(y_resampled)*100:.2f}%)")
        print(f"  Post-SMOTE Ratio:    {after_counts[0]/after_counts[1]:.2f}")

        # Construct resampled DataFrame
        smote_df = pd.DataFrame(X_resampled, columns=available_cols)
        smote_df['is_correct'] = y_resampled
        smote_df.to_csv(PROCESSED_DATA_DIR / "train_smote_flat.csv", index=False)

        metadata = {
            'features': available_cols,
            'before_counts': {int(k): int(v) for k, v in before_counts.items()},
            'after_counts': {int(k): int(v) for k, v in after_counts.items()},
            'total_before': len(y_tr),
            'total_after': len(y_resampled),
            'k_neighbors': self.k_neighbors,
            'source_partition': "train.csv (Development Cohort)",
            'random_seed': self.random_seed
        }

        self._generate_report(metadata)
        return smote_df, metadata

    def _generate_report(self, meta: Dict[str, Any]):
        b0 = meta['before_counts'][0]
        b1 = meta['before_counts'][1]
        a0 = meta['after_counts'][0]
        a1 = meta['after_counts'][1]

        report_lines = [
            "# Synthetic Minority Over-sampling Technique (SMOTE) Report",
            "**Target Partition:** Development Training Set (`train.csv`) Exclusively  ",
            "**Validation Partition (`val.csv`):** 100% UNTOUCHED (Natural Distribution Preserved)  ",
            "**Final Test Partition (`test.csv`):** 100% UNTOUCHED (Strictly Isolated)\n",
            "## 1. Quantitative Class Rebalancing Summary",
            "| Split / Status | Class 0 (Incorrect) | Class 1 (Correct) | Total Samples | Imbalance Ratio (0:1) |",
            "|---|---|---|---|---|",
            f"| **Before SMOTE (train.csv)** | {b0:,} ({b0/meta['total_before']*100:.1f}%) | {b1:,} ({b1/meta['total_before']*100:.1f}%) | {meta['total_before']:,} | {b0/b1:.3f} : 1 |",
            f"| **After SMOTE (train_smote_flat.csv)** | {a0:,} ({a0/meta['total_after']*100:.1f}%) | {a1:,} ({a1/meta['total_after']*100:.1f}%) | {meta['total_after']:,} | 1.00 : 1 |",
            "\n## 2. Academic Defense & Methodological Guarantees",
            "- **Zero Distribution Leakage:** SMOTE synthetic generation was fitted and executed exclusively on the training partition. The validation fold and the 250-student final test cohort remain in their natural un-augmented distribution, preventing synthetic data leakage into evaluation benchmarks.",
            "- **Sequence Causal Integrity:** Sequence models (LSTM, Transformer, BERT-NCF, CNN-LSTM) are trained on authentic chronological student interaction streams; SMOTE is applied to derived tabular decision-boundary representations to prevent temporal graph disruption.",
            f"- **Algorithm Configuration:** Regular SMOTE with $k={meta['k_neighbors']}$ nearest neighbors in feature space (`RandomState={meta['random_seed']}`).",
            f"- **Feature Space ({len(meta['features'])} variables):** `{', '.join(meta['features'])}`."
        ]

        report_path = REPORTS_DIR / "smote_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"  Saved SMOTE report to: {report_path.name}")


class AcademicADASYNHandler:
    """
    Handles academically compliant ADASYN (Adaptive Synthetic Sampling) on training data only.
    ADASYN adaptively generates more synthetic data for minority class examples that are harder
    to learn, based on the density distribution of majority-class neighbors.
    """
    def __init__(self, random_seed: int = RANDOM_SEED, n_neighbors: int = 5):
        self.random_seed = random_seed
        self.n_neighbors = n_neighbors
        from imblearn.over_sampling import ADASYN
        self.adasyn = ADASYN(random_state=random_seed, n_neighbors=n_neighbors)

    def apply_adasyn_to_training(self, train_df: pd.DataFrame,
                                 feature_cols: list = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fits and resamples the training set tabular representation using ADASYN.
        """
        if feature_cols is None:
            feature_cols = [
                'previous_accuracy', 'recent_accuracy_5', 'attempt_count',
                'response_time_norm', 'time_since_prev_norm',
                'source_encoded', 'platform_encoded', 'part', 'num_responses',
                'question_idx'
            ]

        available_cols = [c for c in feature_cols if c in train_df.columns]
        train_clean = train_df.dropna(subset=available_cols + ['is_correct']).copy()

        X_tr = train_clean[available_cols].values
        y_tr = train_clean['is_correct'].values.astype(int)

        before_counts = Counter(y_tr)
        print("\n" + "=" * 75)
        print("  ACADEMIC ADASYN APPLICATION (TRAINING DATA ONLY)")
        print("=" * 75)
        print(f"  Class 0 (Incorrect): {before_counts[0]:>8,d} ({before_counts[0]/len(y_tr)*100:.2f}%)")
        print(f"  Class 1 (Correct):   {before_counts[1]:>8,d} ({before_counts[1]/len(y_tr)*100:.2f}%)")
        print(f"  Initial Imbalance Ratio (0:1): {before_counts[0]/before_counts[1]:.3f}")

        # Resample with ADASYN
        X_resampled, y_resampled = self.adasyn.fit_resample(X_tr, y_tr)
        after_counts = Counter(y_resampled)

        print(f"\n  Resampled Class 0:   {after_counts[0]:>8,d} ({after_counts[0]/len(y_resampled)*100:.2f}%)")
        print(f"  Resampled Class 1:   {after_counts[1]:>8,d} ({after_counts[1]/len(y_resampled)*100:.2f}%)")
        print(f"  Post-ADASYN Ratio:   {after_counts[0]/after_counts[1]:.3f}")

        # Construct resampled DataFrame
        adasyn_df = pd.DataFrame(X_resampled, columns=available_cols)
        adasyn_df['is_correct'] = y_resampled
        output_file = PROCESSED_DATA_DIR / "train_adasyn_flat.csv"
        adasyn_df.to_csv(output_file, index=False)

        metadata = {
            'features': available_cols,
            'before_counts': {int(k): int(v) for k, v in before_counts.items()},
            'after_counts': {int(k): int(v) for k, v in after_counts.items()},
            'total_before': len(y_tr),
            'total_after': len(y_resampled),
            'n_neighbors': self.n_neighbors,
            'source_partition': "train.csv (Development Cohort)",
            'random_seed': self.random_seed
        }

        self._generate_report(metadata)
        return adasyn_df, metadata

    def _generate_report(self, meta: Dict[str, Any]):
        b0 = meta['before_counts'][0]
        b1 = meta['before_counts'][1]
        a0 = meta['after_counts'][0]
        a1 = meta['after_counts'][1]

        report_lines = [
            "# Adaptive Synthetic Sampling (ADASYN) Report",
            "**Target Partition:** Development Training Set (`train.csv`) Exclusively  ",
            "**Validation Partition (`val.csv`):** 100% UNTOUCHED (Natural Distribution Preserved)  ",
            "**Final Test Partition (`test.csv`):** 100% UNTOUCHED (Strictly Isolated)\n",
            "## 1. Quantitative Class Rebalancing Summary",
            "| Split / Status | Class 0 (Incorrect) | Class 1 (Correct) | Total Samples | Imbalance Ratio (0:1) |",
            "|---|---|---|---|---|",
            f"| **Before ADASYN (train.csv)** | {b0:,} ({b0/meta['total_before']*100:.1f}%) | {b1:,} ({b1/meta['total_before']*100:.1f}%) | {meta['total_before']:,} | {b0/b1:.3f} : 1 |",
            f"| **After ADASYN (train_adasyn_flat.csv)** | {a0:,} ({a0/meta['total_after']*100:.1f}%) | {a1:,} ({a1/meta['total_after']*100:.1f}%) | {meta['total_after']:,} | {a0/a1:.3f} : 1 |",
            "\n## 2. Theoretical Mechanism & Academic Justification",
            "- **Adaptive Density Distribution:** Unlike standard SMOTE, which generates synthetic minority examples uniformly along $k$-NN segments, ADASYN uses a density distribution $\\Gamma_i = r_i / \\sum r_i$ where $r_i = \\Delta_i / K$ (the proportion of majority-class examples among $K$ nearest neighbors of minority instance $x_i$).",
            "- **Focus on Boundary Hardness:** More synthetic samples are adaptively generated for minority instances that are harder to learn (those near the ambiguous boundary between correct and incorrect student actions).",
            "- **Zero Distribution Leakage:** ADASYN was fitted and executed exclusively on `train.csv`. Validation (`val.csv`) and Unseen Test (`test.csv`) sets remain in their untouched natural distribution.",
            f"- **Algorithm Hyperparameters:** $K = {meta['n_neighbors']}$ nearest neighbors (`RandomState={meta['random_seed']}`).",
            f"- **Feature Space ({len(meta['features'])} variables):** `{', '.join(meta['features'])}`."
        ]

        report_path = REPORTS_DIR / "adasyn_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"  Saved ADASYN report to: {report_path.name}")


class AcademicResamplingManager:
    """
    Unified manager for class rebalancing strategies:
    - 'none': Untouched natural training distribution
    - 'smote': Standard Synthetic Minority Over-sampling
    - 'adasyn': Adaptive Synthetic Sampling (boundary focus)
    - 'gan': Tabular Conditional WGAN-GP Generative Augmentation
    """
    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.smote_handler = AcademicSMOTEHandler(random_seed=random_seed)
        self.adasyn_handler = AcademicADASYNHandler(random_seed=random_seed)

    def rebalance(self, train_df: pd.DataFrame, method: str = 'smote', **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        method = method.lower().strip()
        if method == 'smote':
            return self.smote_handler.apply_smote_to_training(train_df, **kwargs)
        elif method == 'adasyn':
            return self.adasyn_handler.apply_adasyn_to_training(train_df, **kwargs)
        elif method == 'gan':
            from src.gan_augmentation import TabularWGANGP
            gan = TabularWGANGP(random_seed=self.random_seed)
            return gan.augment_training(train_df, **kwargs)
        elif method in ('none', 'natural'):
            print("  Resampling: Natural distribution preserved (no oversampling).")
            return train_df, {'method': 'none'}
        else:
            raise ValueError(f"Unknown rebalancing method: '{method}'. Choose from 'none', 'smote', 'adasyn', 'gan'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Academic Data Rebalancing (SMOTE / ADASYN / GAN)")
    parser.add_argument("--method", type=str, default="both", choices=["smote", "adasyn", "both", "all"],
                        help="Rebalancing method to apply to train.csv")
    args = parser.parse_args()

    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        print(f"Error: {train_path} not found. Run data pipeline first.")
        sys.exit(1)

    print(f"Loading {train_path}...")
    train_df = pd.read_csv(train_path)

    mgr = AcademicResamplingManager()
    if args.method in ["smote", "both", "all"]:
        mgr.rebalance(train_df, method="smote")
    if args.method in ["adasyn", "both", "all"]:
        mgr.rebalance(train_df, method="adasyn")
    if args.method in ["all"]:
        mgr.rebalance(train_df, method="gan")
