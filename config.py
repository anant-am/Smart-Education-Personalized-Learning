"""
Smart Education Project — Central Configuration
=================================================
All dataset paths, hyperparameters, and project settings are stored here.
Do NOT hard-code paths in individual Python files.
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(r"A:\edge download\Smart_Education_Project")

# Original datasets (READ-ONLY — never modify these)
EDNET_CONTENTS_DIR = Path(r"A:\edge download\EdNet-Contents\contents")
EDNET_KT3_DEV_DIR = Path(r"A:\edge download\EdNet-KT3\KT-3 (1000)")
EDNET_KT3_TEST_DIR = Path(r"A:\edge download\EdNet-KT3\KT-3 (250) TEST")
EDNET_KT3_FULL_DIR = Path(r"A:\edge download\EdNet-KT3\KT3")
EDNET_KT3_DIR = EDNET_KT3_DEV_DIR  # Default development cohort

# Contents files
QUESTIONS_CSV = EDNET_CONTENTS_DIR / "questions.csv"
LECTURES_CSV = EDNET_CONTENTS_DIR / "lectures.csv"
COUPONS_CSV = EDNET_CONTENTS_DIR / "coupons.csv"
PAYMENTS_CSV = EDNET_CONTENTS_DIR / "payments.csv"

# Project output directories (all processed data goes here)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Video + Audio Attention Datasets
VIDEO_AUDIO_DATA_DIR = PROJECT_ROOT / "video audio dataset"
VIDEO_ATTENTION_CSV = VIDEO_AUDIO_DATA_DIR / "video_attention_dataset.csv"
AUDIO_ATTENTION_CSV = VIDEO_AUDIO_DATA_DIR / "audio_attention_dataset.csv"

# Model outputs
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ATTENTION_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "attention"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = PROJECT_ROOT / "plots"

EXPERIMENT_MANIFEST_PATH = REPORTS_DIR / "experiment_manifest.json"
COMPLIANCE_CHECKLIST_PATH = REPORTS_DIR / "final_compliance_checklist.md"

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Hardware / Training Defaults
# ============================================================
# Starting conservative values for RTX 3050 6GB VRAM
DEFAULT_SEQUENCE_LENGTH = 50
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3

# Mixed precision
USE_MIXED_PRECISION = True

# ============================================================
# Development progression
# ============================================================
DEBUG_USER_COUNT = 20       # For debugging
SMALL_EXPERIMENT_USERS = 100
MEDIUM_EXPERIMENT_USERS = 500
FULL_EXPERIMENT_USERS = 1000

# ============================================================
# Ensure output directories exist
# ============================================================
def ensure_dirs():
    """Create project output directories if they don't exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
              MODELS_DIR, CHECKPOINTS_DIR, ATTENTION_CHECKPOINTS_DIR,
              LOGS_DIR, REPORTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Ollama Generative AI Layer
# ============================================================
try:
    from ollama_ai.ollama_config import (
        OLLAMA_ENABLED,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        OLLAMA_TIMEOUT,
        OLLAMA_TEMPERATURE,
        OLLAMA_MAX_TOKENS,
        get_ollama_config,
        is_ollama_enabled,
    )
except ImportError:
    OLLAMA_ENABLED = False
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:3b"
    OLLAMA_TIMEOUT = 35
    OLLAMA_TEMPERATURE = 0.7
    OLLAMA_MAX_TOKENS = 1024

