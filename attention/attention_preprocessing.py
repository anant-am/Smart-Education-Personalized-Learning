"""
Smart Education Project — Attention Data Preprocessing & Leakage Auditor
========================================================================
Handles:
1. Academic Leakage Verification:
   - Target leakage audit (strictly bans General_Class and Target from inputs)
   - Exact duplicate row detection
   - Preprocessing isolation (scalers fitted exclusively on training partitions)
2. Programmatic Stratified Splitting:
   - 80% Development Cohort (1,600 samples)
   - 20% Isolated Unseen Test Cohort (400 samples)
3. Scaler Fitting & Serialization:
   - Independent StandardScalers for video and audio features
   - Artifact persistence to models/checkpoints/attention/attention_preprocessor.pkl
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
PROJECT_ROOT = Path(r"A:\edge download\Smart_Education_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_config import (
    VIDEO_FEATURE_NAMES,
    AUDIO_FEATURE_NAMES,
    ATTENTION_LABEL_MAP,
    VIDEO_ATTENTION_CSV,
    AUDIO_ATTENTION_CSV,
    ATTENTION_CHECKPOINTS_DIR,
    ATTENTION_PREPROCESSOR_PATH,
    AttentionConfig
)


class AttentionPreprocessor:
    """
    Orchestrates academic-grade preprocessing and feature engineering for
    multimodal attention detection, guaranteeing zero data leakage.
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self.video_scaler = StandardScaler()
        self.audio_scaler = StandardScaler()
        self.is_fitted = False
        self.leakage_report = {}

    def load_and_audit_data(
        self,
        video_csv_path: Optional[Path] = None,
        audio_csv_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads video and audio attention CSVs and performs rigorous leakage and integrity audits.
        """
        v_path = video_csv_path or VIDEO_ATTENTION_CSV
        a_path = audio_csv_path or AUDIO_ATTENTION_CSV

        if not Path(v_path).exists():
            raise FileNotFoundError(f"Video attention dataset not found at {v_path}")
        if not Path(a_path).exists():
            raise FileNotFoundError(f"Audio attention dataset not found at {a_path}")

        video_df = pd.read_csv(v_path)
        audio_df = pd.read_csv(a_path)

        # 1. Column Integrity Check
        missing_v_cols = [c for c in VIDEO_FEATURE_NAMES if c not in video_df.columns]
        missing_a_cols = [c for c in AUDIO_FEATURE_NAMES if c not in audio_df.columns]
        assert not missing_v_cols, f"Missing video features in CSV: {missing_v_cols}"
        assert not missing_a_cols, f"Missing audio features in CSV: {missing_a_cols}"

        # 2. Duplicate Detection
        v_dups = int(video_df[VIDEO_FEATURE_NAMES].duplicated().sum())
        a_dups = int(audio_df[AUDIO_FEATURE_NAMES].duplicated().sum())

        # 3. Target Leakage Verification (CRITICAL: General_Class must NEVER be an input)
        forbidden_inputs = {"General_Class", "Target"}
        v_leaks = [c for c in VIDEO_FEATURE_NAMES if c in forbidden_inputs]
        a_leaks = [c for c in AUDIO_FEATURE_NAMES if c in forbidden_inputs]
        assert len(v_leaks) == 0, f"FATAL: Target leakage detected in video features: {v_leaks}"
        assert len(a_leaks) == 0, f"FATAL: Target leakage detected in audio features: {a_leaks}"

        self.leakage_report = {
            "video_rows": len(video_df),
            "audio_rows": len(audio_df),
            "video_duplicates": v_dups,
            "audio_duplicates": a_dups,
            "general_class_excluded": True,
            "target_excluded": True,
            "target_leakage_status": "PASS (Zero Leaked Target Columns)"
        }

        return video_df, audio_df

    def split_data_programmatically(
        self,
        video_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        test_size: float = 0.20
    ) -> Dict[str, np.ndarray]:
        """
        Executes programmatic stratified 80/20 train/test split.
        Zero manual user file manipulation required.
        """
        # Feature matrices
        X_video = video_df[VIDEO_FEATURE_NAMES].values.astype(np.float32)
        X_audio = audio_df[AUDIO_FEATURE_NAMES].values.astype(np.float32)

        # Authoritative target (from video dataset)
        y = video_df["Target"].values.astype(np.int64)

        # Stratified 80/20 split
        indices = np.arange(len(y))
        dev_idx, test_idx, y_dev, y_test = train_test_split(
            indices,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.config.random_seed
        )

        return {
            "X_video_dev": X_video[dev_idx],
            "X_audio_dev": X_audio[dev_idx],
            "y_dev": y_dev,
            "X_video_test": X_video[test_idx],
            "X_audio_test": X_audio[test_idx],
            "y_test": y_test,
            "dev_indices": dev_idx,
            "test_indices": test_idx
        }

    def fit_and_transform_dev(
        self,
        X_video_dev: np.ndarray,
        X_audio_dev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits scalers exclusively on Development partition.
        """
        X_v_norm = self.video_scaler.fit_transform(X_video_dev)
        X_a_norm = self.audio_scaler.fit_transform(X_audio_dev)
        self.is_fitted = True
        return X_v_norm, X_a_norm

    def transform_test(
        self,
        X_video_test: np.ndarray,
        X_audio_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms test partition using frozen scalers without refitting.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot transform test set: Preprocessor has not been fitted on Development data.")
        X_v_norm = self.video_scaler.transform(X_video_test)
        X_a_norm = self.audio_scaler.transform(X_audio_test)
        return X_v_norm, X_a_norm

    def transform_single_video_features(self, feat_vec: np.ndarray) -> np.ndarray:
        """Transforms a 10-D video feature vector for real-time inference."""
        feat_2d = np.asarray(feat_vec, dtype=np.float32).reshape(1, -1)
        return self.video_scaler.transform(feat_2d)[0]

    def transform_single_audio_features(self, feat_vec: np.ndarray) -> np.ndarray:
        """Transforms a 10-D audio feature vector for real-time inference."""
        feat_2d = np.asarray(feat_vec, dtype=np.float32).reshape(1, -1)
        return self.audio_scaler.transform(feat_2d)[0]

    def save_artifacts(self, save_path: Optional[Path] = None):
        """Serializes fitted scalers and metadata."""
        out_path = save_path or ATTENTION_PREPROCESSOR_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_scaler": self.video_scaler,
            "audio_scaler": self.audio_scaler,
            "video_feature_names": VIDEO_FEATURE_NAMES,
            "audio_feature_names": AUDIO_FEATURE_NAMES,
            "label_map": ATTENTION_LABEL_MAP,
            "is_fitted": self.is_fitted,
            "leakage_report": self.leakage_report
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [OK] Preprocessing artifacts saved to {out_path}")

    @classmethod
    def load_artifacts(cls, load_path: Optional[Path] = None) -> "AttentionPreprocessor":
        """Loads fitted preprocessor from saved artifact file."""
        in_path = load_path or ATTENTION_PREPROCESSOR_PATH
        if not Path(in_path).exists():
            raise FileNotFoundError(f"Attention preprocessor artifact not found at {in_path}")
        with open(in_path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj.video_scaler = payload["video_scaler"]
        obj.audio_scaler = payload["audio_scaler"]
        obj.is_fitted = payload.get("is_fitted", True)
        obj.leakage_report = payload.get("leakage_report", {})
        return obj
