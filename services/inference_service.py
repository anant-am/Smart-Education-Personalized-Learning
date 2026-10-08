"""
Smart Education — Inference Service Layer for Streamlit GUI
============================================================
Wraps MultiModelInferenceEngine from main.py with Streamlit-compatible
caching and structured data access. Models are loaded ONCE and persist
across page navigations via @st.cache_resource.
"""

import sys
import os
import json
import time
import io
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_DATA_DIR, CHECKPOINTS_DIR, REPORTS_DIR, PLOTS_DIR,
    MODELS_DIR, PROJECT_ROOT as CFG_ROOT
)


@st.cache_resource(show_spinner="Loading 5 trained models into GPU memory...")
def get_inference_engine():
    """
    Singleton-cached MultiModelInferenceEngine.
    Loads all 5 PyTorch checkpoints ONCE and keeps them in GPU/CPU memory.
    """
    # Suppress engine's print output during GUI loading
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from main import MultiModelInferenceEngine
        engine = MultiModelInferenceEngine()
    finally:
        sys.stdout = old_stdout
    return engine


@st.cache_data(ttl=3600)
def get_experiment_manifest() -> Dict[str, Any]:
    """Loads the experiment manifest JSON."""
    path = REPORTS_DIR / "experiment_manifest.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def get_model_results(model_num: int) -> Dict[str, Any]:
    """Loads model training results JSON for a specific model."""
    path = REPORTS_DIR / f"model{model_num}_results.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
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
def get_checkpoint_metadata() -> Dict[str, Dict[str, Any]]:
    """Loads _meta.json files for all checkpoints."""
    meta = {}
    for f in CHECKPOINTS_DIR.rglob("*_meta.json"):
        with open(f, 'r', encoding='utf-8') as fh:
            meta[f.stem.replace('_meta', '')] = json.load(fh)
    return meta


@st.cache_data(ttl=3600)
def get_checkpoint_files() -> List[Dict[str, Any]]:
    """Lists all checkpoint files with size and modification time."""
    files = []
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
def get_attention_results() -> Dict[str, Any]:
    """Loads multimodal attention training and evaluation results JSON."""
    path = REPORTS_DIR / "attention_results.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def get_augmentation_comparison() -> Optional[pd.DataFrame]:
    """Loads augmentation comparison CSV."""
    path = REPORTS_DIR / "augmentation_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=3600)
def get_five_model_comparison() -> Optional[pd.DataFrame]:
    """Loads the 5-model comparison CSV."""
    path = REPORTS_DIR / "five_model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=3600)
def get_cv_results(model_num: int) -> Optional[pd.DataFrame]:
    """Loads cross-validation fold-by-fold CSV for a model."""
    # Try multiple naming conventions
    patterns = [
        f"model{model_num}_cv_10fold.csv",
        f"model_cv_10fold.csv",
    ]
    # Also try descriptive names
    model_names = {
        1: "model_1_(lstm+attention)_cv_10fold.csv",
        2: "model_2_(transformer+kt)_cv_10fold.csv",
        3: "model_3_(bert-ncf)_cv_10fold.csv",
        5: "model_5_(cnn+lstm)_cv_10fold.csv",
    }
    if model_num in model_names:
        patterns.insert(0, model_names[model_num])

    for p in patterns:
        path = REPORTS_DIR / p
        if path.exists():
            return pd.read_csv(path)
    return None


@st.cache_data(ttl=3600)
def get_report_markdown(filename: str) -> str:
    """Loads a markdown report as text."""
    path = REPORTS_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


@st.cache_data(ttl=3600)
def get_dataset_stats() -> Dict[str, Any]:
    """Computes dataset statistics from CSV files."""
    stats = {}
    for split in ['train', 'val', 'test']:
        path = PROCESSED_DATA_DIR / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path, nrows=0)  # just columns
            # Get row count efficiently
            with open(path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1  # minus header
            stats[split] = {
                'rows': row_count,
                'columns': list(df.columns),
                'num_columns': len(df.columns)
            }
    return stats


@st.cache_data(ttl=3600)
def get_dataset_sample(split: str = "test", n_rows: int = 100) -> Optional[pd.DataFrame]:
    """Loads a sample of rows from a dataset split."""
    path = PROCESSED_DATA_DIR / f"{split}.csv"
    if path.exists():
        return pd.read_csv(path, nrows=n_rows)
    return None


@st.cache_data(ttl=3600)
def get_dataset_user_counts() -> Dict[str, int]:
    """Counts unique users per split."""
    counts = {}
    for split in ['train', 'val', 'test']:
        path = PROCESSED_DATA_DIR / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=['user_id'])
            counts[split] = df['user_id'].nunique()
    return counts


def get_plot_path(filename: str) -> Optional[Path]:
    """Returns the path to a plot file if it exists."""
    path = PLOTS_DIR / filename
    return path if path.exists() else None


def get_available_plots() -> List[str]:
    """Lists all available plot PNG files."""
    if PLOTS_DIR.exists():
        return sorted([f.name for f in PLOTS_DIR.glob("*.png")])
    return []


def predict_student(engine, student_id: str) -> Dict[str, Any]:
    """Wraps engine.predict_student_by_id with timing."""
    t0 = time.time()
    # Suppress engine prints during GUI inference
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = engine.predict_student_by_id(student_id)
    finally:
        sys.stdout = old_stdout
    if result:
        result['total_time_ms'] = (time.time() - t0) * 1000
    return result


def predict_custom(engine, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wraps engine.predict_sequence with timing."""
    t0 = time.time()
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = engine.predict_sequence(events)
    finally:
        sys.stdout = old_stdout
    result['total_time_ms'] = (time.time() - t0) * 1000
    result['student_id'] = 'Custom Simulated Student'
    return result


def get_system_info(engine) -> Dict[str, Any]:
    """Gathers system hardware and software information."""
    import torch
    info = {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(engine.device) if engine is not None else ('cuda' if torch.cuda.is_available() else 'cpu'),
        'project_root': str(CFG_ROOT),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info['cuda_version'] = torch.version.cuda or 'N/A'
    try:
        import streamlit
        info['streamlit_version'] = streamlit.__version__
    except Exception:
        info['streamlit_version'] = 'N/A'
    try:
        import platform
        info['os'] = platform.platform()
        info['cpu'] = platform.processor()
    except Exception:
        pass
    return info
