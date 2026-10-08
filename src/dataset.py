"""
Smart Education Project — PyTorch Dataset Module
=================================================
Provides:
1. KTSequenceDataset: Chronological sequence dataset for Knowledge Tracing
   (LSTM+Attention, Transformer+KT, BERT-NCF, CNN-LSTM).
   Strict next-event target prediction with zero-padding masks.
2. RecommenderDataset: Sparse / dense interaction profile dataset for
   Autoencoder + Recommender network.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional


class KTSequenceDataset(Dataset):
    """
    Chronological Sequence Dataset for Knowledge Tracing Models.
    Each item represents a student's interaction history:
    - Input features: interaction details at step t (k <= t)
    - Prediction target: correctness of the *next* interaction at step t+1
    - Causal integrity: No future outcomes leak into earlier steps
    - Masking: Non-events / padding positions have mask = False
    """

    def __init__(self, sequences: List[List[Dict[str, Any]]], max_seq_len: int = 50,
                 user_ids: Optional[List[str]] = None):
        """
        Args:
            sequences: List of student sequences. Each sequence is a list of event dicts.
            max_seq_len: Maximum sequence length. Longer sequences keep the most recent interactions.
            user_ids: Optional list of student identifiers corresponding to each sequence.
        """
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.user_ids = user_ids if user_ids is not None else [f"student_{i}" for i in range(len(sequences))]
        self.num_features = 9  # q_id, part, tag1, tag2, tag3, is_correct, resp_time, source, platform

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        uid = self.user_ids[idx] if idx < len(self.user_ids) else ""

        features = np.zeros((self.max_seq_len, self.num_features), dtype=np.float32)
        targets = np.zeros((self.max_seq_len,), dtype=np.float32)
        mask = np.zeros((self.max_seq_len,), dtype=bool)

        # Edge case: Sequence has fewer than 2 events -> cannot form next-event pair
        if len(seq) < 2:
            return (
                torch.FloatTensor(features),
                torch.FloatTensor(targets),
                torch.BoolTensor(mask),
                {"seq_len": 0, "user_id": uid}
            )

        # Truncate to the most recent max_seq_len + 1 interactions
        if len(seq) > self.max_seq_len + 1:
            seq = seq[-(self.max_seq_len + 1):]

        seq_len = len(seq) - 1  # Number of transitions: t -> t+1

        for i in range(seq_len):
            event = seq[i]
            next_event = seq[i + 1]

            # Tags (pad with 0 to 3 concepts)
            tags = event.get('tags', [])
            tag1 = int(tags[0]) if len(tags) > 0 else 0
            tag2 = int(tags[1]) if len(tags) > 1 else 0
            tag3 = int(tags[2]) if len(tags) > 2 else 0

            features[i] = [
                float(event.get('question_id', 0)),
                float(event.get('part', 0)),
                float(tag1),
                float(tag2),
                float(tag3),
                float(event.get('is_correct', 0)),
                float(event.get('response_time', 0.0)),
                float(event.get('source_encoded', 0)),
                float(event.get('platform_encoded', 0))
            ]

            # Target is the outcome of the subsequent question
            targets[i] = float(next_event.get('is_correct', 0))
            mask[i] = True

        return (
            torch.FloatTensor(features),
            torch.FloatTensor(targets),
            torch.BoolTensor(mask),
            {"seq_len": seq_len, "user_id": uid}
        )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader batching."""
        features, targets, masks, metadata = zip(*batch)
        return (
            torch.stack(features),
            torch.stack(targets),
            torch.stack(masks),
            metadata
        )


class RecommenderDataset(Dataset):
    """
    Interaction Profile Dataset for Autoencoder + Recommender.
    Processes student-resource interaction vectors.
    """

    def __init__(self, interaction_matrix, labels=None, user_ids: Optional[List[str]] = None):
        if hasattr(interaction_matrix, 'toarray'):
            self.interactions = interaction_matrix.toarray().astype(np.float32)
        else:
            self.interactions = np.array(interaction_matrix, dtype=np.float32)

        self.labels = np.array(labels, dtype=np.float32) if labels is not None else np.zeros(len(self.interactions), dtype=np.float32)
        self.user_ids = user_ids if user_ids is not None else [f"student_{i}" for i in range(len(self.interactions))]

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int):
        return (
            torch.FloatTensor(self.interactions[idx]),
            torch.FloatTensor([self.labels[idx]]),
            self.user_ids[idx]
        )
