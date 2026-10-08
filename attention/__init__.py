"""
Attention Module — Video + Audio Attention Detection & Personalization
========================================================================
Academic multimodal attention modeling subsystem for the Smart Education Project.
Provides:
- VideoFeatureExtractor: Extracts 10-D facial, gaze, motion, and posture cues from raw video (.mp4/.avi/.mov/.mkv).
- AudioFeatureExtractor: Extracts 10-D speech, energy, pause, and rate cues from audio tracks.
- MultimodalAttentionModel: Deep fusion architecture with unimodal fallbacks and missing modality handling.
- AttentionPreprocessor: Strict leakage prevention, programmatic splitting, and training-only scaling.
- AttentionTrainer: AMP-accelerated training, overfit/underfit diagnosis, fresh model retraining.
- AttentionEvaluator: 10-Fold Stratified CV with training-only SMOTE, untouched test evaluation.
- AttentionInferenceEngine: Fast inference on video files or feature vectors.
- run_video_attention_pipeline: End-to-end integration bridging attention with EdNet knowledge state and recommendations.
"""

from attention.attention_config import (
    VIDEO_FEATURE_NAMES,
    AUDIO_FEATURE_NAMES,
    ATTENTION_LABEL_MAP,
    ATTENTION_CLASS_NAMES,
    AttentionConfig
)
from attention.attention_model import (
    MultimodalAttentionModel,
    VideoEncoder,
    AudioEncoder,
    MultimodalFusion
)
from attention.attention_dataset import AttentionDataset
from attention.attention_preprocessing import AttentionPreprocessor
from attention.attention_trainer import AttentionTrainer
from attention.attention_evaluator import AttentionEvaluator
from attention.video_feature_extractor import VideoFeatureExtractor
from attention.audio_feature_extractor import AudioFeatureExtractor
from attention.attention_inference import AttentionInferenceEngine
from attention.attention_pipeline import run_video_attention_pipeline

__all__ = [
    "VIDEO_FEATURE_NAMES",
    "AUDIO_FEATURE_NAMES",
    "ATTENTION_LABEL_MAP",
    "ATTENTION_CLASS_NAMES",
    "AttentionConfig",
    "MultimodalAttentionModel",
    "VideoEncoder",
    "AudioEncoder",
    "MultimodalFusion",
    "AttentionDataset",
    "AttentionPreprocessor",
    "AttentionTrainer",
    "AttentionEvaluator",
    "VideoFeatureExtractor",
    "AudioFeatureExtractor",
    "AttentionInferenceEngine",
    "run_video_attention_pipeline",
]
