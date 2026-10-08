"""
Smart Education Project — Central Configuration
=================================================
All dataset paths, hyperparameters, and project settings are stored here.
Do NOT hard-code paths in individual Python files.
"""

import os
from pathlib import Path

# ============================================================
# Project Paths & Environment Overrides
# ============================================================
PROJECT_ROOT = Path(os.environ.get("SMART_EDU_PROJECT_ROOT", os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent))))

# Optional .env loader (no external dependency needed)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip().strip("'\"")
                    if _k and _k not in os.environ and _v:
                        os.environ[_k] = _v
    except Exception:
        pass

# Execution Modes: 'smoke', 'dev', 'full'
EXECUTION_MODE = os.environ.get("SMART_EDU_MODE", os.environ.get("EXECUTION_MODE", "dev")).lower().strip()

# Original datasets (READ-ONLY — configurable via environment variables)
_env_contents = os.environ.get("EDNET_CONTENTS_ROOT", os.environ.get("EDNET_CONTENTS_DIR"))
if _env_contents:
    EDNET_CONTENTS_DIR = Path(_env_contents)
else:
    if (PROJECT_ROOT / "data" / "contents").exists():
        EDNET_CONTENTS_DIR = PROJECT_ROOT / "data" / "contents"
    elif (PROJECT_ROOT.parent / "EdNet-Contents" / "contents").exists():
        EDNET_CONTENTS_DIR = PROJECT_ROOT.parent / "EdNet-Contents" / "contents"
    elif (PROJECT_ROOT / "EdNet-Contents" / "contents").exists():
        EDNET_CONTENTS_DIR = PROJECT_ROOT / "EdNet-Contents" / "contents"
    elif Path(r"A:\edge download\EdNet-Contents\contents").exists():
        EDNET_CONTENTS_DIR = Path(r"A:\edge download\EdNet-Contents\contents")
    else:
        EDNET_CONTENTS_DIR = PROJECT_ROOT / "data" / "contents"

_env_full = os.environ.get("EDNET_KT3_ROOT", os.environ.get("EDNET_KT3_FULL_DIR", os.environ.get("EDNET_KT3_DIR")))
if _env_full:
    EDNET_KT3_FULL_DIR = Path(_env_full)
else:
    if (PROJECT_ROOT / "data" / "KT3").exists():
        EDNET_KT3_FULL_DIR = PROJECT_ROOT / "data" / "KT3"
    elif (PROJECT_ROOT.parent / "EdNet-KT3" / "KT3").exists():
        EDNET_KT3_FULL_DIR = PROJECT_ROOT.parent / "EdNet-KT3" / "KT3"
    elif (PROJECT_ROOT / "EdNet-KT3" / "KT3").exists():
        EDNET_KT3_FULL_DIR = PROJECT_ROOT / "EdNet-KT3" / "KT3"
    elif Path(r"A:\edge download\EdNet-KT3\KT3").exists():
        EDNET_KT3_FULL_DIR = Path(r"A:\edge download\EdNet-KT3\KT3")
    else:
        EDNET_KT3_FULL_DIR = PROJECT_ROOT / "data" / "KT3"

_default_dev = Path(r"A:\edge download\EdNet-KT3\KT-3 (1000)")
_default_test = Path(r"A:\edge download\EdNet-KT3\KT-3 (250) TEST")
EDNET_KT3_DEV_DIR = Path(os.environ.get("EDNET_KT3_DEV_DIR", str(_default_dev if _default_dev.exists() else EDNET_KT3_FULL_DIR)))
EDNET_KT3_TEST_DIR = Path(os.environ.get("EDNET_KT3_TEST_DIR", str(_default_test if _default_test.exists() else EDNET_KT3_FULL_DIR)))
EDNET_KT3_DIR = EDNET_KT3_FULL_DIR if EXECUTION_MODE == "full" else EDNET_KT3_DEV_DIR

# Contents files
QUESTIONS_CSV = EDNET_CONTENTS_DIR / "questions.csv"
LECTURES_CSV = EDNET_CONTENTS_DIR / "lectures.csv"
COUPONS_CSV = EDNET_CONTENTS_DIR / "coupons.csv"
PAYMENTS_CSV = EDNET_CONTENTS_DIR / "payments.csv"

# Project output directories (all processed data goes here)
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", os.environ.get("SMART_EDU_OUTPUT_DIR", str(PROJECT_ROOT))))
DATA_DIR = Path(os.environ.get("SMART_EDU_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_DIR = Path(os.environ.get("SMART_EDU_RAW_DATA_DIR", str(DATA_DIR / "raw")))
PROCESSED_DATA_DIR = Path(os.environ.get("PROCESSED_DATA_ROOT", os.environ.get("SMART_EDU_PROCESSED_DATA_DIR", str(DATA_DIR / "processed"))))

# Video + Audio Attention Datasets
VIDEO_AUDIO_DATA_DIR = Path(os.environ.get("VIDEO_AUDIO_DATA_DIR", str(PROJECT_ROOT / "video audio dataset")))
VIDEO_ATTENTION_CSV = VIDEO_AUDIO_DATA_DIR / "video_attention_dataset.csv"
AUDIO_ATTENTION_CSV = VIDEO_AUDIO_DATA_DIR / "audio_attention_dataset.csv"

# Model outputs
MODELS_DIR = Path(os.environ.get("SMART_EDU_MODELS_DIR", str(OUTPUT_ROOT / "models" if OUTPUT_ROOT != PROJECT_ROOT else PROJECT_ROOT / "models")))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINT_ROOT", os.environ.get("SMART_EDU_CHECKPOINTS_DIR", os.environ.get("CHECKPOINTS_DIR", str(MODELS_DIR / "checkpoints")))))
ATTENTION_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "attention"
LOGS_DIR = Path(os.environ.get("SMART_EDU_LOGS_DIR", str(OUTPUT_ROOT / "logs" if OUTPUT_ROOT != PROJECT_ROOT else PROJECT_ROOT / "logs")))
REPORTS_DIR = Path(os.environ.get("SMART_EDU_REPORTS_DIR", str(OUTPUT_ROOT / "reports" if OUTPUT_ROOT != PROJECT_ROOT else PROJECT_ROOT / "reports")))
FIGURES_DIR = REPORTS_DIR / "figures"
PLOTS_DIR = Path(os.environ.get("SMART_EDU_PLOTS_DIR", str(OUTPUT_ROOT / "plots" if OUTPUT_ROOT != PROJECT_ROOT else PROJECT_ROOT / "plots")))

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
              LOGS_DIR, REPORTS_DIR, FIGURES_DIR, PLOTS_DIR]:
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

