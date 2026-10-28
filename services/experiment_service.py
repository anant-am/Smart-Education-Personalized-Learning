# -*- coding: utf-8 -*-
"""
Smart Education — Experiment Results & Benchmarks Service Layer
===============================================================
Dedicated service for loading, caching, and formatting empirical experiment results,
hyperparameter matrices, benchmark comparisons, and diagnostic figure paths.

Separates experiment evaluation data loading from live model inference services.
Handles missing files and missing columns gracefully with zero data fabrication.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    REPORTS_DIR,
    PLOTS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
    ATTENTION_CHECKPOINTS_DIR,
    MODELS_DIR,
)

# Standard schema expected for the 40-configuration experimental matrix
EXPECTED_EXPERIMENT_COLUMNS = [
    'run_id', 'model_key', 'model_name', 'val_strategy', 'aug_condition',
    'parameters', 'train_users', 'val_users', 'test_users',
    'train_events', 'val_events', 'test_events',
    'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc', 'val_pr_auc', 'val_log_loss',
    'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_pr_auc', 'test_log_loss',
    'train_time_sec', 'inference_time_sec', 'throughput_events_per_sec',
    'peak_ram_mb', 'peak_vram_mb', 'fit_status', 'correction_applied',
    'before_val_auc', 'after_val_auc', 'checkpoint_path', 'cv_metrics',
    'timestamp', 'status',
    'reconstruction_mse', 'hitrate_at_5', 'ndcg_at_5', 'precision_at_5', 'recall_at_5'
]


STANDARD_MODELS = [
    ('model1', 'Model 1: LSTM + Attention'),
    ('model2', 'Model 2: Transformer + Knowledge Tracing'),
    ('model3', 'Model 3: BERT-style Transformer Encoder + NCF'),
    ('model4', 'Model 4: Autoencoder + Recommender Network'),
    ('model5', 'Model 5: CNN + LSTM'),
]
STANDARD_AUGS = ['baseline', 'smote', 'adasyn', 'gan']
STANDARD_VALS = ['no_10fold', '10fold']


@st.cache_data(ttl=3600)
def get_canonical_manifest() -> Dict[str, Any]:
    """Loads reports/canonical_results_manifest.json."""
    path = REPORTS_DIR / "canonical_results_manifest.json"
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=3600)
def get_experiment_results_df() -> pd.DataFrame:
    """
    Loads authentic experimental results from the canonical results source.
    Primary source: reports/canonical_results.csv
    Fallback source: reports/experiment_results.csv

    Strictly returns only authentic saved configurations without fabricating placeholder rows.
    """
    canonical_path = REPORTS_DIR / "canonical_results.csv"
    fallback_path = REPORTS_DIR / "experiment_results.csv"

    if canonical_path.exists() and canonical_path.stat().st_size > 0:
        try:
            return pd.read_csv(canonical_path)
        except Exception:
            pass

    if fallback_path.exists() and fallback_path.stat().st_size > 0:
        try:
            return pd.read_csv(fallback_path)
        except Exception:
            pass

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_five_model_comparison() -> Optional[pd.DataFrame]:
    """Loads the 5-model comparison CSV (reports/five_model_comparison.csv)."""
    path = REPORTS_DIR / "five_model_comparison.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(ttl=3600)
def get_final_comparison() -> Optional[pd.DataFrame]:
    """Loads final_comparison.csv (reports/final_comparison.csv)."""
    path = REPORTS_DIR / "final_comparison.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(ttl=3600)
def get_augmentation_comparison() -> Optional[pd.DataFrame]:
    """Loads augmentation comparison CSV (reports/augmentation_comparison.csv)."""
    path = REPORTS_DIR / "augmentation_comparison.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(ttl=3600)
def get_experiment_manifest() -> Dict[str, Any]:
    """Loads experiment_manifest.json (reports/experiment_manifest.json)."""
    path = REPORTS_DIR / "experiment_manifest.json"
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=3600)
def get_model_results(model_num: int) -> Dict[str, Any]:
    """Loads model training results JSON for a specific model."""
    path = REPORTS_DIR / f"model{model_num}_results.json"
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=3600)
def get_all_model_results() -> Dict[int, Dict[str, Any]]:
    """Loads results for all 5 models."""
    results = {}
    for i in range(1, 6):
        r = get_model_results(i)
        if r:
            results[i] = r
    return results


@st.cache_data(ttl=3600)
def get_checkpoint_files() -> List[Dict[str, Any]]:
    """Lists all checkpoint files with size and modification time."""
    files = []
    if CHECKPOINTS_DIR.exists():
        for f in sorted(CHECKPOINTS_DIR.rglob("*")):
            if f.is_file():
                stat = f.stat()
                rel_name = f.relative_to(CHECKPOINTS_DIR)
                files.append({
                    'name': str(rel_name),
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime)),
                    'extension': f.suffix
                })
    return files


@st.cache_data(ttl=3600)
def get_checkpoint_metadata() -> Dict[str, Dict[str, Any]]:
    """Loads _meta.json files for all checkpoints."""
    meta = {}
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.rglob("*_meta.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    meta[f.stem.replace('_meta', '')] = json.load(fh)
            except Exception:
                pass
    return meta


@st.cache_data(ttl=3600)
def get_attention_results() -> Dict[str, Any]:
    """Loads multimodal attention training and evaluation results JSON."""
    path = REPORTS_DIR / "attention_results.json"
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_figure_path(filename: str) -> Optional[Path]:
    """Returns the path to a figure file in reports/figures or plots/ if it exists."""
    path = FIGURES_DIR / filename
    if path.exists():
        return path
    alt = PLOTS_DIR / filename
    return alt if alt.exists() else None


def get_available_figures() -> List[str]:
    """Lists all available figure PNG files in reports/figures."""
    if FIGURES_DIR.exists():
        return sorted([f.name for f in FIGURES_DIR.glob("*.png")])
    return []


def get_plot_path(filename: str) -> Optional[Path]:
    """Returns the path to a plot file in plots/ or reports/figures if it exists."""
    path = PLOTS_DIR / filename
    if path.exists():
        return path
    alt = FIGURES_DIR / filename
    return alt if alt.exists() else None


def get_available_plots() -> List[str]:
    """Lists all available plot PNG files."""
    if PLOTS_DIR.exists():
        return sorted([f.name for f in PLOTS_DIR.glob("*.png")])
    return []
