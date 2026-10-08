"""
Smart Education Project — Attention Module Configuration
=========================================================
Central configuration for Video + Audio Attention modeling, feature definitions,
target mappings, and hyperparameters.
"""

from pathlib import Path
from dataclasses import dataclass
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import RANDOM_SEED, CHECKPOINTS_DIR, REPORTS_DIR, PLOTS_DIR, VIDEO_AUDIO_DATA_DIR

# Feature definitions according to authoritative dataset headers
VIDEO_FEATURE_NAMES = [
    "Face_Visibility",
    "Eye_Gaze_Score",
    "Head_Pose_Stability",
    "Blink_Rate",
    "Facial_Expression_Intensity",
    "Body_Posture_Score",
    "Movement_Level",
    "Focus_Duration",
    "Screen_Orientation",
    "Attention_Score"
]

AUDIO_FEATURE_NAMES = [
    "Speech_Presence",
    "Voice_Energy",
    "Speech_Clarity",
    "Background_Noise_Level",
    "Speaking_Rate",
    "Pause_Frequency",
    "Response_Delay",
    "Interaction_Frequency",
    "Audio_Engagement_Score",
    "Listening_Consistency"
]

# Multiclass target mappings
ATTENTION_LABEL_MAP = {
    0: "Inattentive",
    1: "Partially Attentive",
    2: "Attentive"
}

ATTENTION_CLASS_NAMES = ["Inattentive", "Partially Attentive", "Attentive"]

# Paths
VIDEO_ATTENTION_CSV = VIDEO_AUDIO_DATA_DIR / "video_attention_dataset.csv"
AUDIO_ATTENTION_CSV = VIDEO_AUDIO_DATA_DIR / "audio_attention_dataset.csv"
ATTENTION_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "attention"
ATTENTION_REPORT_PATH = REPORTS_DIR / "attention_report.md"
ATTENTION_RESULTS_JSON = REPORTS_DIR / "attention_results.json"
ATTENTION_PREPROCESSOR_PATH = ATTENTION_CHECKPOINTS_DIR / "attention_preprocessor.pkl"


@dataclass
class AttentionConfig:
    """Hyperparameter and operational configuration for multimodal attention model."""
    video_dim: int = 10
    audio_dim: int = 10
    hidden_dim: int = 64
    latent_dim: int = 32
    num_classes: int = 3
    dropout: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 15
    patience: int = 4
    random_seed: int = RANDOM_SEED
    modality_dropout: float = 0.15  # For robust missing-modality handling during training
    use_mixed_precision: bool = True
