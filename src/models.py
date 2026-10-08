"""
Smart Education Project — Standardized Deep Learning Model Architectures
========================================================================
Standardized Numbering (Mandatory Academic University Specification):
  MODEL 1 — LSTM + Attention (Knowledge Tracing)
  MODEL 2 — Transformer + Knowledge Tracing
  MODEL 3 — BERT-style Transformer + Neural Collaborative Filtering (NCF)
  MODEL 4 — Autoencoder + Recommender Network
  MODEL 5 — CNN + LSTM

All models support:
- Causal interaction masking (no future lookahead)
- Model-derived knowledge state extraction per concept/tag
- Xavier/Kaiming initialization
- Memory-efficient execution on NVIDIA RTX 3050 6GB VRAM
- Mixed precision and CUDA acceleration
"""

import math
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for sequence Transformer architectures."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# MODEL 1: LSTM + Attention Knowledge Tracing
# ============================================================================
class LSTMAttentionKT(nn.Module):
    """
    MODEL 1: LSTM + Attention for Knowledge Tracing.
    Fuses question embeddings, concept tag embeddings, part representations,
    and continuous features through an LSTM layer followed by a causal
    self-attention mechanism to predict next-step response correctness.
    """
    def __init__(self, num_questions: int = 13170, num_parts: int = 8, num_tags: int = 300,
                 embed_dim: int = 64, hidden_dim: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, num_cont_features: int = 6):
        super(LSTMAttentionKT, self).__init__()
        self.num_tags = num_tags
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Item and Categorical Embeddings (0 is reserved for padding/unknown)
        self.q_emb = nn.Embedding(num_questions + 2, embed_dim, padding_idx=0)
        self.part_emb = nn.Embedding(num_parts + 2, embed_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(num_tags + 2, embed_dim, padding_idx=0)
        self.cont_proj = nn.Linear(num_cont_features, embed_dim)

        # Multi-feature fusion projection
        self.input_proj = nn.Linear(embed_dim * 4, embed_dim)

        # Recurrent layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Causal Attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Binary classification head for next-response prediction logits
        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, inputs: torch.Tensor, return_attention: bool = False):
        """
        Args:
            inputs: Tensor of shape [batch_size, seq_len, 9]
        Returns:
            logits: Tensor of shape [batch_size, seq_len]
        """
        q_ids = inputs[:, :, 0].long().clamp(min=0, max=self.q_emb.num_embeddings - 1)
        part_ids = inputs[:, :, 1].long().clamp(min=0, max=self.part_emb.num_embeddings - 1)
        tag_ids = inputs[:, :, 2].long().clamp(min=0, max=self.tag_emb.num_embeddings - 1)
        cont_feats = inputs[:, :, 3:].float()

        q_e = self.q_emb(q_ids)
        part_e = self.part_emb(part_ids)
        tag_e = self.tag_emb(tag_ids)
        cont_e = self.cont_proj(cont_feats)

        x = torch.cat([q_e, part_e, tag_e, cont_e], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)

        lstm_out, _ = self.lstm(x)

        # Causal Self-Attention
        Q = self.q_proj(lstm_out)
        K = self.k_proj(lstm_out)
        V = self.v_proj(lstm_out)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        seq_len = lstm_out.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)

        logits = self.out_fc(attn_out).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits

    def compute_knowledge_state(self, inputs: torch.Tensor, mask: torch.Tensor) -> Dict[int, float]:
        """
        Computes genuine model-informed concept mastery state from student sequence.
        Aggregates predicted probabilities for questions tagged with each concept.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(inputs)
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        mask_np = mask.cpu().numpy()

        concept_scores = {}
        concept_counts = {}

        for b in range(inputs_np.shape[0]):
            for t in range(inputs_np.shape[1]):
                if not mask_np[b, t]:
                    continue
                # Extract tags: tag1, tag2, tag3 at index 2, 3, 4
                tag_candidates = [int(inputs_np[b, t, 2]), int(inputs_np[b, t, 3]), int(inputs_np[b, t, 4])]
                valid_tags = [t_id for t_id in tag_candidates if t_id > 0]
                pred_p = float(probs_np[b, t])

                for tag in valid_tags:
                    concept_scores[tag] = concept_scores.get(tag, 0.0) + pred_p
                    concept_counts[tag] = concept_counts.get(tag, 0) + 1

        mastery_state = {}
        for tag, total in concept_scores.items():
            mastery_state[tag] = total / concept_counts[tag]
        return mastery_state


# ============================================================================
# MODEL 2: Transformer + Knowledge Tracing
# ============================================================================
class TransformerKT(nn.Module):
    """
    MODEL 2: Transformer + Knowledge Tracing.
    Uses multi-head self-attention with strict upper-triangular causal masking
    to trace learner mastery transitions across chronological interactions.
    """
    def __init__(self, num_questions: int = 13170, num_parts: int = 8, num_tags: int = 300,
                 embed_dim: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.2, max_seq_len: int = 50,
                 num_cont_features: int = 6):
        super(TransformerKT, self).__init__()
        self.num_tags = num_tags
        self.embed_dim = embed_dim

        self.q_emb = nn.Embedding(num_questions + 2, embed_dim, padding_idx=0)
        self.part_emb = nn.Embedding(num_parts + 2, embed_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(num_tags + 2, embed_dim, padding_idx=0)
        self.cont_proj = nn.Linear(num_cont_features, embed_dim)

        self.input_proj = nn.Linear(embed_dim * 4, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q_ids = inputs[:, :, 0].long().clamp(min=0, max=self.q_emb.num_embeddings - 1)
        part_ids = inputs[:, :, 1].long().clamp(min=0, max=self.part_emb.num_embeddings - 1)
        tag_ids = inputs[:, :, 2].long().clamp(min=0, max=self.tag_emb.num_embeddings - 1)
        cont_feats = inputs[:, :, 3:].float()

        q_e = self.q_emb(q_ids)
        part_e = self.part_emb(part_ids)
        tag_e = self.tag_emb(tag_ids)
        cont_e = self.cont_proj(cont_feats)

        x = torch.cat([q_e, part_e, tag_e, cont_e], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.pos_encoder(x)

        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        memory = self.transformer_encoder(x, mask=causal_mask, is_causal=True)

        logits = self.out_fc(memory).squeeze(-1)
        return logits

    def compute_knowledge_state(self, inputs: torch.Tensor, mask: torch.Tensor) -> Dict[int, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(inputs)
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        mask_np = mask.cpu().numpy()

        concept_scores = {}
        concept_counts = {}

        for b in range(inputs_np.shape[0]):
            for t in range(inputs_np.shape[1]):
                if not mask_np[b, t]:
                    continue
                tag_candidates = [int(inputs_np[b, t, 2]), int(inputs_np[b, t, 3]), int(inputs_np[b, t, 4])]
                valid_tags = [t_id for t_id in tag_candidates if t_id > 0]
                pred_p = float(probs_np[b, t])

                for tag in valid_tags:
                    concept_scores[tag] = concept_scores.get(tag, 0.0) + pred_p
                    concept_counts[tag] = concept_counts.get(tag, 0) + 1

        return {tag: total / concept_counts[tag] for tag, total in concept_scores.items()}


# ============================================================================
# MODEL 3: BERT-style Transformer + Neural Collaborative Filtering (BERT + NCF)
# ============================================================================
class BERTNCF(nn.Module):
    """
    MODEL 3: BERT-style Transformer + Neural Collaborative Filtering.
    Employs a BERT-style Transformer encoder to derive learner state vectors,
    fused with target item embeddings through Generalized Matrix Factorization (GMF)
    and Multi-Layer Perceptron (MLP) branches into a NeuMF collaborative prediction.
    """
    def __init__(self, num_questions: int = 13170, num_parts: int = 8, num_tags: int = 300,
                 embed_dim: int = 64, nhead: int = 4, num_layers: int = 2,
                 mlp_dims: list = [128, 64], dropout: float = 0.2, max_seq_len: int = 50,
                 num_cont_features: int = 6):
        super(BERTNCF, self).__init__()
        self.num_tags = num_tags
        self.embed_dim = embed_dim

        self.q_emb = nn.Embedding(num_questions + 2, embed_dim, padding_idx=0)
        self.part_emb = nn.Embedding(num_parts + 2, embed_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(num_tags + 2, embed_dim, padding_idx=0)
        self.cont_proj = nn.Linear(num_cont_features, embed_dim)

        self.input_proj = nn.Linear(embed_dim * 4, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len + 10)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.bert_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Target Item representation for collaborative filtering
        self.target_q_emb = nn.Embedding(num_questions + 2, embed_dim, padding_idx=0)

        # NCF: Multi-Layer Perceptron (MLP) branch
        mlp_layers = []
        in_dim = embed_dim * 2
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # NeuMF final prediction head (combining GMF embed_dim and MLP final dim)
        self.prediction_head = nn.Linear(embed_dim + mlp_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q_ids = inputs[:, :, 0].long().clamp(min=0, max=self.q_emb.num_embeddings - 1)
        part_ids = inputs[:, :, 1].long().clamp(min=0, max=self.part_emb.num_embeddings - 1)
        tag_ids = inputs[:, :, 2].long().clamp(min=0, max=self.tag_emb.num_embeddings - 1)
        cont_feats = inputs[:, :, 3:].float()

        q_e = self.q_emb(q_ids)
        part_e = self.part_emb(part_ids)
        tag_e = self.tag_emb(tag_ids)
        cont_e = self.cont_proj(cont_feats)

        x = torch.cat([q_e, part_e, tag_e, cont_e], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.pos_encoder(x)

        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        student_reps = self.bert_encoder(x, mask=causal_mask, is_causal=True)

        target_e = self.target_q_emb(q_ids)

        # GMF Branch: element-wise interaction
        gmf = student_reps * target_e

        # MLP Branch: concatenated non-linear interaction
        mlp_in = torch.cat([student_reps, target_e], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # NeuMF fusion
        neumf = torch.cat([gmf, mlp_out], dim=-1)
        logits = self.prediction_head(neumf).squeeze(-1)
        return logits

    def compute_knowledge_state(self, inputs: torch.Tensor, mask: torch.Tensor) -> Dict[int, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(inputs)
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        mask_np = mask.cpu().numpy()

        concept_scores = {}
        concept_counts = {}

        for b in range(inputs_np.shape[0]):
            for t in range(inputs_np.shape[1]):
                if not mask_np[b, t]:
                    continue
                tag_candidates = [int(inputs_np[b, t, 2]), int(inputs_np[b, t, 3]), int(inputs_np[b, t, 4])]
                valid_tags = [t_id for t_id in tag_candidates if t_id > 0]
                pred_p = float(probs_np[b, t])

                for tag in valid_tags:
                    concept_scores[tag] = concept_scores.get(tag, 0.0) + pred_p
                    concept_counts[tag] = concept_counts.get(tag, 0) + 1

        return {tag: total / concept_counts[tag] for tag, total in concept_scores.items()}


# ============================================================================
# MODEL 4: Autoencoder + Recommender Network
# ============================================================================
class AutoencoderRecommender(nn.Module):
    """
    MODEL 4: Autoencoder + Recommender Network.
    Compresses student interaction profile into a compact latent bottleneck,
    reconstructs historical engagement, and scores candidate learning resources
    (lectures, explanations, practice questions) through a dedicated recommender head.
    """
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 64,
                 num_resources: int = 500, dropout: float = 0.2):
        super(AutoencoderRecommender, self).__init__()
        self.input_dim = input_dim
        self.num_resources = num_resources

        # Encoder Network: Compresses interaction profile
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder Network: Reconstructs interaction vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Recommendation Head: Predicts relevance across learning resources
        self.recommender = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_resources)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstructed: [batch_size, input_dim]
            resource_scores: [batch_size, num_resources]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        resource_scores = self.recommender(latent)
        return reconstructed, resource_scores

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def compute_knowledge_state(self, x: torch.Tensor, resource_to_tags: Optional[Dict[int, List[int]]] = None) -> Dict[int, float]:
        """
        Computes model-derived concept mastery/relevance state from Autoencoder recommendations.
        Aggregates predicted resource relevance probabilities across mapped concepts.
        """
        self.eval()
        with torch.no_grad():
            _, rec_scores = self.forward(x)
            probs = torch.sigmoid(rec_scores).cpu().numpy()

        concept_scores = {}
        concept_counts = {}

        if resource_to_tags is None:
            num_res = probs.shape[1]
            resource_to_tags = {r_idx: [(r_idx % 50) + 1] for r_idx in range(num_res)}

        for b in range(probs.shape[0]):
            for r_idx in range(probs.shape[1]):
                tags = resource_to_tags.get(r_idx, [])
                score = float(probs[b, r_idx])
                for t in tags:
                    if t > 0:
                        concept_scores[t] = concept_scores.get(t, 0.0) + score
                        concept_counts[t] = concept_counts.get(t, 0) + 1

        return {tag: total / concept_counts[tag] for tag, total in concept_scores.items()}


# ============================================================================
# MODEL 5: CNN + LSTM
# ============================================================================
class CNNLSTM(nn.Module):
    """
    MODEL 5: CNN + LSTM Hybrid Architecture.
    Extracts localized multi-step sequential practice patterns using 1D temporal
    convolutions, which are fed into an LSTM layer to track long-term cumulative
    learning trajectories and predict next-response outcomes.
    """
    def __init__(self, num_questions: int = 13170, num_parts: int = 8, num_tags: int = 300,
                 embed_dim: int = 64, hidden_dim: int = 64, kernel_size: int = 3,
                 num_layers: int = 1, dropout: float = 0.2, num_cont_features: int = 6):
        super(CNNLSTM, self).__init__()
        self.num_tags = num_tags
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.q_emb = nn.Embedding(num_questions + 2, embed_dim, padding_idx=0)
        self.part_emb = nn.Embedding(num_parts + 2, embed_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(num_tags + 2, embed_dim, padding_idx=0)
        self.cont_proj = nn.Linear(num_cont_features, embed_dim)

        self.input_proj = nn.Linear(embed_dim * 4, embed_dim)

        # 1D Temporal Convolution (extracts local practice features)
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Recurrent Layer for long-term trajectory modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Binary classification output head
        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q_ids = inputs[:, :, 0].long().clamp(min=0, max=self.q_emb.num_embeddings - 1)
        part_ids = inputs[:, :, 1].long().clamp(min=0, max=self.part_emb.num_embeddings - 1)
        tag_ids = inputs[:, :, 2].long().clamp(min=0, max=self.tag_emb.num_embeddings - 1)
        cont_feats = inputs[:, :, 3:].float()

        q_e = self.q_emb(q_ids)
        part_e = self.part_emb(part_ids)
        tag_e = self.tag_emb(tag_ids)
        cont_e = self.cont_proj(cont_feats)

        x = torch.cat([q_e, part_e, tag_e, cont_e], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)

        # Conv1D requires [Batch, Channels, Length]
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.dropout(x_conv)

        # Transpose back to [Batch, Length, Channels] for LSTM
        x_lstm = x_conv.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm)

        logits = self.out_fc(lstm_out).squeeze(-1)
        return logits

    def compute_knowledge_state(self, inputs: torch.Tensor, mask: torch.Tensor) -> Dict[int, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(inputs)
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        mask_np = mask.cpu().numpy()

        concept_scores = {}
        concept_counts = {}

        for b in range(inputs_np.shape[0]):
            for t in range(inputs_np.shape[1]):
                if not mask_np[b, t]:
                    continue
                tag_candidates = [int(inputs_np[b, t, 2]), int(inputs_np[b, t, 3]), int(inputs_np[b, t, 4])]
                valid_tags = [t_id for t_id in tag_candidates if t_id > 0]
                pred_p = float(probs_np[b, t])

                for tag in valid_tags:
                    concept_scores[tag] = concept_scores.get(tag, 0.0) + pred_p
                    concept_counts[tag] = concept_counts.get(tag, 0) + 1

        return {tag: total / concept_counts[tag] for tag, total in concept_scores.items()}
