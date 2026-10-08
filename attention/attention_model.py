"""
Smart Education Project — Multimodal Video + Audio Attention Model
===================================================================
Academic Multimodal Deep Learning Architecture:
1. Video Encoder: Non-linear MLP projecting 10-D video features into a normalized latent space.
2. Audio Encoder: Non-linear MLP projecting 10-D audio features into a normalized latent space.
3. Multimodal Fusion: Gated cross-modal projection layer combining latent modalities with
   dynamic availability masks (m_v, m_a).
4. Unimodal Robustness: Gracefully operates in:
   - Video-only mode (missing audio)
   - Audio-only mode (missing video / no face)
   - Full multimodal mode (both video and audio present)
5. Multiclass Attention Head: Computes 3-class logits, softmax probabilities, and prediction confidence.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_config import ATTENTION_LABEL_MAP, AttentionConfig


class VideoEncoder(nn.Module):
    """Encodes 10-D visual attention features into a compact representation."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.2):
        super(VideoEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioEncoder(nn.Module):
    """Encodes 10-D acoustic attention features into a compact representation."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.2):
        super(AudioEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusion(nn.Module):
    """
    Fuses video and audio latent representations with modality availability masking.
    Supports missing modalities gracefully without NaN propagation or crashes.
    """
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2):
        super(MultimodalFusion, self).__init__()
        # Input: [h_v (32) + h_a (32) + mask_v (1) + mask_a (1)] = 66
        fusion_in_dim = (latent_dim * 2) + 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(
        self,
        h_v: torch.Tensor,
        h_a: torch.Tensor,
        mask_v: torch.Tensor,
        mask_a: torch.Tensor
    ) -> torch.Tensor:
        # Reshape masks to [batch_size, 1]
        m_v = mask_v.float().view(-1, 1)
        m_a = mask_a.float().view(-1, 1)

        # Zero out latent vectors if modality is missing
        h_v_masked = h_v * m_v
        h_a_masked = h_a * m_a

        # Concatenate latent embeddings with modality presence indicators
        combined = torch.cat([h_v_masked, h_a_masked, m_v, m_a], dim=1)
        fused = self.fusion_mlp(combined)
        return fused


class MultimodalAttentionModel(nn.Module):
    """
    Complete Multimodal Video + Audio Attention Neural Network.
    Predicts 3 attention states:
      0: Inattentive
      1: Partially Attentive
      2: Attentive
    """
    def __init__(
        self,
        video_dim: int = 10,
        audio_dim: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.25
    ):
        super(MultimodalAttentionModel, self).__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.video_encoder = VideoEncoder(video_dim, hidden_dim, latent_dim, dropout)
        self.audio_encoder = AudioEncoder(audio_dim, hidden_dim, latent_dim, dropout)
        self.fusion = MultimodalFusion(latent_dim, hidden_dim, dropout)

        # Multiclass attention classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        mask_v: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_v: Video features [B, 10]
            x_a: Audio features [B, 10]
            mask_v: BoolTensor [B] indicating video presence (defaults to all True)
            mask_a: BoolTensor [B] indicating audio presence (defaults to all True)
        Returns:
            logits: FloatTensor [B, 3]
        """
        device = x_v.device
        batch_size = x_v.size(0)

        if mask_v is None:
            mask_v = torch.ones(batch_size, dtype=torch.bool, device=device)
        if mask_a is None:
            mask_a = torch.ones(batch_size, dtype=torch.bool, device=device)

        h_v = self.video_encoder(x_v)
        h_a = self.audio_encoder(x_a)

        h_fused = self.fusion(h_v, h_a, mask_v, mask_a)
        logits = self.classifier(h_fused)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        mask_v: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Inference forward pass returning predicted class, confidence, and probability distribution.
        """
        self.eval()
        logits = self.forward(x_v, x_a, mask_v, mask_a)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        label_name = ATTENTION_LABEL_MAP.get(pred_idx, f"Class_{pred_idx}")

        prob_dict = {
            ATTENTION_LABEL_MAP.get(c, f"Class_{c}"): float(probs[c])
            for c in range(self.num_classes)
        }

        return {
            "attention_class": label_name,
            "class_index": pred_idx,
            "confidence": conf,
            "probabilities": prob_dict,
            "logits": logits[0].cpu().numpy().tolist()
        }
