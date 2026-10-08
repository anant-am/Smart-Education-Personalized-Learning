"""
Smart Education Project — Attention Dataset
============================================
PyTorch Dataset representation for Video + Audio Attention multimodal modeling.
Supports paired multimodal features, unimodal tensors, and dynamic modality masking
to guarantee robust inference when audio or video is missing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


class AttentionDataset(Dataset):
    """
    Multimodal Dataset for Attention Classification.
    Provides:
    - video_features: FloatTensor of shape [N, 10]
    - audio_features: FloatTensor of shape [N, 10]
    - video_mask: BoolTensor [N] indicating video availability
    - audio_mask: BoolTensor [N] indicating audio availability
    - targets: LongTensor [N] (0: Inattentive, 1: Partially Attentive, 2: Attentive)
    """

    def __init__(
        self,
        video_features: np.ndarray,
        audio_features: np.ndarray,
        targets: np.ndarray,
        video_mask: Optional[np.ndarray] = None,
        audio_mask: Optional[np.ndarray] = None,
        modality_dropout: float = 0.0,
        is_training: bool = False
    ):
        self.video_features = np.asarray(video_features, dtype=np.float32)
        self.audio_features = np.asarray(audio_features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)

        n_samples = len(self.targets)
        assert len(self.video_features) == n_samples, "Video features and targets length mismatch"
        assert len(self.audio_features) == n_samples, "Audio features and targets length mismatch"

        if video_mask is None:
            self.video_mask = np.ones((n_samples,), dtype=bool)
        else:
            self.video_mask = np.asarray(video_mask, dtype=bool)

        if audio_mask is None:
            self.audio_mask = np.ones((n_samples,), dtype=bool)
        else:
            self.audio_mask = np.asarray(audio_mask, dtype=bool)

        self.modality_dropout = modality_dropout
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v_feat = self.video_features[idx].copy()
        a_feat = self.audio_features[idx].copy()
        v_m = bool(self.video_mask[idx])
        a_m = bool(self.audio_mask[idx])
        target = int(self.targets[idx])

        # Apply training-time modality dropout to train robust unimodal fallbacks
        if self.is_training and self.modality_dropout > 0.0:
            rand_val = np.random.rand()
            if rand_val < self.modality_dropout:
                # Drop audio modality
                a_feat = np.zeros_like(a_feat)
                a_m = False
            elif rand_val < 2 * self.modality_dropout:
                # Drop video modality
                v_feat = np.zeros_like(v_feat)
                v_m = False

        return (
            torch.from_numpy(v_feat).float(),
            torch.from_numpy(a_feat).float(),
            torch.tensor(v_m, dtype=torch.bool),
            torch.tensor(a_m, dtype=torch.bool),
            torch.tensor(target, dtype=torch.long)
        )
