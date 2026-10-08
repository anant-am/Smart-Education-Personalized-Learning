"""
Smart Education Project — Attention Inference Engine
=====================================================
Unified inference layer executing real-time attention prediction from:
1. Actual video files (.mp4, .avi, .mov, .mkv)
2. Tabular feature vectors (benchmarking & unit tests)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch

import sys
PROJECT_ROOT = Path(r"A:\edge download\Smart_Education_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_config import (
    ATTENTION_CHECKPOINTS_DIR,
    ATTENTION_LABEL_MAP,
    AttentionConfig
)
from attention.attention_model import MultimodalAttentionModel
from attention.attention_preprocessing import AttentionPreprocessor
from attention.video_feature_extractor import VideoFeatureExtractor
from attention.audio_feature_extractor import AudioFeatureExtractor


class AttentionInferenceEngine:
    """
    Hosts the pre-trained Multimodal Attention Model and feature extractors.
    Loads checkpoints ONCE into GPU/CPU memory for instant forward passes.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = AttentionConfig()
        self.preprocessor = AttentionPreprocessor.load_artifacts()

        # Initialize feature extractors
        self.video_extractor = VideoFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()

        # Load model weights
        self._load_model()

    def _load_model(self):
        ckpt_path = ATTENTION_CHECKPOINTS_DIR / "attention_multimodal_corrected_best.pth"
        if not ckpt_path.exists():
            ckpt_path = ATTENTION_CHECKPOINTS_DIR / "attention_multimodal_initial_best.pth"
        if not ckpt_path.exists():
            # Fallback to any .pth in attention checkpoints
            pths = list(ATTENTION_CHECKPOINTS_DIR.glob("*.pth"))
            if pths:
                ckpt_path = pths[0]
            else:
                raise FileNotFoundError(f"No attention checkpoints found in {ATTENTION_CHECKPOINTS_DIR}")

        self.model = MultimodalAttentionModel(
            video_dim=self.config.video_dim,
            audio_dim=self.config.audio_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_classes=self.config.num_classes,
            dropout=0.0  # Eval mode
        ).to(self.device)

        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.active_checkpoint = ckpt_path.name
        print(f"  [OK] AttentionInferenceEngine active: Loaded {self.active_checkpoint} on {self.device}")

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Executes end-to-end inference directly on a raw video file.
        Extracts video and audio features, transforms with saved scalers,
        and computes multimodal attention predictions.
        """
        v_res = self.video_extractor.extract_features_from_video(video_path)
        a_res = self.audio_extractor.extract_features_from_video(video_path)

        all_warnings = v_res.get("warnings", []) + a_res.get("warnings", [])
        v_avail = v_res.get("video_available", False)
        a_avail = a_res.get("audio_available", False)

        # Check if both modalities are completely absent/unreadable
        if not v_avail and not a_avail:
            return {
                "status": "Insufficient visual and acoustic information",
                "attention_class": "Indeterminate",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in ATTENTION_LABEL_MAP.values()},
                "video_available": False,
                "audio_available": False,
                "warnings": all_warnings + ["Neither face cues nor audio signals could be extracted"],
                "video_features": v_res.get("features", {}),
                "audio_features": a_res.get("features", {}),
                "checkpoint": self.active_checkpoint
            }

        # Normalize features with frozen development scalers
        v_scaled = self.preprocessor.transform_single_video_features(v_res["feature_vector"])
        a_scaled = self.preprocessor.transform_single_audio_features(a_res["feature_vector"])

        t_v = torch.from_numpy(v_scaled).unsqueeze(0).to(self.device)
        t_a = torch.from_numpy(a_scaled).unsqueeze(0).to(self.device)
        m_v = torch.tensor([v_avail], dtype=torch.bool, device=self.device)
        m_a = torch.tensor([a_avail], dtype=torch.bool, device=self.device)

        pred_res = self.model.predict(t_v, t_a, m_v, m_a)

        return {
            "status": "Success",
            "attention_class": pred_res["attention_class"],
            "class_index": pred_res["class_index"],
            "confidence": pred_res["confidence"],
            "probabilities": pred_res["probabilities"],
            "video_available": v_avail,
            "audio_available": a_avail,
            "warnings": all_warnings,
            "video_features": v_res["features"],
            "audio_features": a_res["features"],
            "video_duration_sec": v_res.get("duration_sec", 0.0),
            "checkpoint": self.active_checkpoint
        }

    def predict_features(
        self,
        video_features: np.ndarray,
        audio_features: np.ndarray,
        video_available: bool = True,
        audio_available: bool = True
    ) -> Dict[str, Any]:
        """
        Executes prediction on raw (unscaled) 10-D feature vectors.
        """
        v_scaled = self.preprocessor.transform_single_video_features(video_features)
        a_scaled = self.preprocessor.transform_single_audio_features(audio_features)

        t_v = torch.from_numpy(v_scaled).unsqueeze(0).to(self.device)
        t_a = torch.from_numpy(a_scaled).unsqueeze(0).to(self.device)
        m_v = torch.tensor([video_available], dtype=torch.bool, device=self.device)
        m_a = torch.tensor([audio_available], dtype=torch.bool, device=self.device)

        pred_res = self.model.predict(t_v, t_a, m_v, m_a)
        pred_res["video_available"] = video_available
        pred_res["audio_available"] = audio_available
        pred_res["checkpoint"] = self.active_checkpoint
        return pred_res
