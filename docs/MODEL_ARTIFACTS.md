# Model Artifacts Specification & Checkpoint Registry

## 1. Executive Summary

This document provides the definitive technical specification and verification registry for all neural network checkpoints preserved in the **Smart Education — AI-Based Personalized Learning Recommendation System**.

Every model checkpoint described herein was trained offline on EdNet-KT3 and multimodal learner attention datasets. All weights have been independently audited, SHA-256 checksummed, loaded into memory via PyTorch, and verified against the initial baseline hashes recorded in `BASELINE_ARTIFACT_HASHES.json`.

> [!IMPORTANT]
> **Strict Non-Destructive Preservation Policy**: Under no circumstances should checkpoints be retrained, fine-tuned, overwritten, or renamed in place. All runtime inference consumes these authentic, frozen weights.

---

## 2. Model Checkpoint Inventory

### Summary Table

| Model ID | Architecture Name | Active Checkpoint File | Checkpoint Size | Parameter Count | Primary Task |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | LSTM + Attention KT | `model1_lstm_corrected_best.pth` | 3,220,213 bytes | 803,201 | Sequential Knowledge Tracing |
| **Model 2** | Transformer + KT (SAKT) | `model2_transformer_corrected_best.pth` | 3,327,995 bytes | 824,385 | Non-Local Sequence Knowledge Tracing |
| **Model 3** | BERT-Style + NCF | `model3_bert_ncf_corrected_best.pth` | 6,322,041 bytes | 1,572,737 | Collaborative Student-Item Interaction |
| **Model 4** | Autoencoder Recommender | `model4_autoencoder_corrected_best.pth` | 419,091 bytes | 104,500 | Concept Profile Mastery & Item Ranking |
| **Model 5** | Causal CNN + LSTM KT | `model5_cnn_lstm_corrected_best.pth` | 3,221,097 bytes | 803,425 | Causal Temporal Feature Extraction |
| **Attention** | Multimodal Fusion Network | `attention/attention_multimodal_corrected_best.pth` | 66,311 bytes | 15,811 | Visual + Acoustic Engagement Classification |

---

## 3. Detailed Architecture & Tensor Specifications

### Model 1: LSTM + Attention KT
- **Checkpoint Path:** `models/checkpoints/model1_lstm_corrected_best.pth`
- **Sidecar Metadata:** `models/checkpoints/model1_lstm_corrected_best_meta.json`
- **SHA-256 Checksum:** `4241eb1de2ef1ddbd8f2fee6f8a475d2fbab8e05e24b6880a72a99d8d5ccea0c`
- **Number of Parameters:** 803,201
- **Architecture Configuration:**
  - Embedding Dimensions: Question Embeddings (128-D), Part Embeddings (32-D), Tag Embeddings (32-D)
  - Hidden Dimension: 128
  - Recurrent Layers: 2-layer Bidirectional LSTM
  - Attention Mechanism: Multi-Head Bahdanau Sequence Attention (4 heads)
  - Output Layer: Linear Projection `[128 -> 1]` with Sigmoid activation
- **Input Contract:**
  - Shape: `[batch_size, sequence_length, 9]`
  - Features: `[question_id, part, tag1, tag2, tag3, correctness, elapsed_time_norm, source, platform]`
- **Output Contract:**
  - Shape: `[batch_size, sequence_length, 1]` representing predicted correctness probability $\hat{y}_t \in [0, 1]$

---

### Model 2: Transformer + Knowledge Tracing (SAKT)
- **Checkpoint Path:** `models/checkpoints/model2_transformer_corrected_best.pth`
- **Sidecar Metadata:** `models/checkpoints/model2_transformer_corrected_best_meta.json`
- **SHA-256 Checksum:** `945a6adda689e091cf4d14f24693b7987e3312a5de14a09d118581f39fd27ec2`
- **Number of Parameters:** 824,385
- **Architecture Configuration:**
  - Embedding Dimensions: 128-D item & exercise embeddings + Sinusoidal Positional Encoding
  - Transformer Layers: 2 Self-Attention Encoder Layers
  - Attention Heads: 4 Attention Heads per layer
  - Feed-Forward Dimension ($d_{ff}$): 512 with GELU activation and LayerNorm
  - Dropout: 0.1
- **Input Contract:**
  - Shape: `[batch_size, sequence_length, 9]`
- **Output Contract:**
  - Shape: `[batch_size, sequence_length, 1]` representing step-wise correctness predictions

---

### Model 3: BERT-Style Transformer Encoder + Neural Collaborative Filtering (NCF)
- **Checkpoint Path:** `models/checkpoints/model3_bert_ncf_corrected_best.pth`
- **Sidecar Metadata:** `models/checkpoints/model3_bert_ncf_corrected_best_meta.json`
- **SHA-256 Checksum:** `0fc640e4453531fbf5233ea647ea374c43ee0f70f6d2bca8148b28cf3b44b823`
- **Number of Parameters:** 1,572,737
- **Architecture Configuration:**
  - Transformer Encoder: 4-layer Bidirectional Transformer Backbone (Hidden Size: 128)
  - GMF Head: Generalized Matrix Factorization latent dot product
  - MLP Head: 3-layer Dense Network (`[256 -> 128 -> 64]`)
  - Fusion Layer: Concatenated NeuMF output `[64 + 64 -> 1]`
- **Input Contract:**
  - Shape: `[batch_size, sequence_length, 9]`
- **Output Contract:**
  - Shape: `[batch_size, 1]` or `[batch_size, sequence_length, 1]` latent collaborative prediction score

---

### Model 4: Autoencoder Recommender Network
- **Checkpoint Path:** `models/checkpoints/model4_autoencoder_corrected_best.pth`
- **Sidecar Metadata:** `models/checkpoints/model4_autoencoder_corrected_best_meta.json`
- **SHA-256 Checksum:** `2b711a319f94bc560258df3ebacba26ac98e1d84a0535a6fd58a7abc191da6aa`
- **Number of Parameters:** 104,500
- **Architecture Configuration:**
  - Input Dimension: 500 (Learner Concept Mastery Vector over active question concepts)
  - Encoder: `Linear(500, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 32)`
  - Latent Representation: 32-D dense learner representation
  - Decoder: `Linear(32, 64) -> ReLU -> Linear(64, 128) -> ReLU -> Linear(128, 500) -> Sigmoid`
- **Task & Evaluation Metrics:**
  - Task: Profile reconstruction and top-$K$ exercise ranking
  - Hit Rate@5: **47.30%**
  - NDCG@5: **0.1571**
  - Reconstruction Error (MSE): **0.1095**
- **Input Contract:**
  - Shape: `[batch_size, 500]` binary or continuous mastery profile vector
- **Output Contract:**
  - Shape: `[batch_size, 500]` reconstructed mastery/preference score vector for ranking

---

### Model 5: Causal CNN + LSTM Knowledge Tracing
- **Checkpoint Path:** `models/checkpoints/model5_cnn_lstm_corrected_best.pth`
- **Sidecar Metadata:** `models/checkpoints/model5_cnn_lstm_corrected_best_meta.json`
- **SHA-256 Checksum:** `15b81bf166046e7f22572a15998dfc38413693fe4c2b92ab3c40333215904d9c`
- **Number of Parameters:** 803,425
- **Architecture Configuration:**
  - 1D Temporal Convolutional Layer: Kernel size 3, 64 filters, Left-Padding only (Strictly Causal)
  - Recurrent Layer: 1-layer LSTM with 128 hidden units
  - Dense Classifier: `Linear(128, 64) -> Dropout(0.2) -> Linear(64, 1) -> Sigmoid`
- **Audit & Causal Verification Note:**
  - Enforces left-sided causal padding (`padding=(kernel_size - 1, 0)`) so the model cannot peek at future questions $t+1 \dots T$.
  - Authenticated Complete-Dataset Test Accuracy: **0.6228 (62.28%)**.
  - Confirmed legacy figure `0.9884` was produced prior to causal padding correction due to bidirectional temporal leakage.
- **Input Contract:**
  - Shape: `[batch_size, sequence_length, 9]`
- **Output Contract:**
  - Shape: `[batch_size, sequence_length, 1]`

---

### Multimodal Attention Fusion Network
- **Checkpoint Path:** `models/checkpoints/attention/attention_multimodal_corrected_best.pth`
- **Sidecar Metadata:** `models/checkpoints/attention/attention_multimodal_corrected_best_meta.json`
- **Scaler / Preprocessor:** `models/checkpoints/attention/attention_preprocessor.pkl`
- **SHA-256 Checksum:** `a9d593003ebcb5e9a4f48b8cf47bfb3cb4f8d55a29ae706788db3d8aa536b6bc`
- **Number of Parameters:** 15,811
- **Architecture Configuration:**
  - Visual Subnet: `Linear(10, 32) -> ReLU -> BatchNorm1d(32)`
  - Acoustic Subnet: `Linear(10, 32) -> ReLU -> BatchNorm1d(32)`
  - Multimodal Attention Fusion: Multi-Head Cross-Modal Attention (`embed_dim=64, num_heads=2`)
  - Classification Head: `Linear(64, 32) -> Dropout(0.3) -> Linear(32, 3) -> Softmax`
- **Class Labels:**
  - `0`: Inattentive
  - `1`: Partially Attentive
  - `2`: Attentive
- **Input Contract:**
  - Concatenated 20-D feature vector `[batch_size, 20]`: 10-D visual + 10-D audio metrics.
- **Output Contract:**
  - Class probabilities `[p_inattentive, p_partially_attentive, p_attentive]` and assigned state string.

---

## 4. Preprocessing & Normalization Artifacts

| Artifact Name | Path | Size | Description |
| :--- | :--- | :--- | :--- |
| **Master Metadata** | `data/processed/metadata.pkl` | 129,572 bytes | Item mappings, vocabulary sizes, max sequence lengths, tag lookup tables. |
| **Questions Catalog** | `data/processed/questions_contents.csv` | 275,695 bytes | 13,169 verified items with `question_id`, `part`, `tags`. |
| **Lectures Catalog** | `data/processed/lectures_contents.csv` | 23,286 bytes | 1,021 video lectures with `lecture_id`, `part`, `tag`. |
| **Sample Test Set** | `data/processed/sample_test_students.csv` | 338,893 bytes | 11,445 interaction rows across 100 test students. |
| **User Splits** | `data/processed/split_users.json` | 13,362 bytes | Clean user-level split partition hashes. |
| **Attention Scaler** | `models/checkpoints/attention/attention_preprocessor.pkl` | 1,779 bytes | Scikit-learn StandardScaler fitted on 20-D feature distributions. |

---

## 5. Checkpoint Verification Protocol

The integrity of all checkpoints is programmatically verified using the regression suite:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_attention_regression.py -v
```

All 21 checkpoints in `models/checkpoints/` have been verified bit-for-bit against `BASELINE_ARTIFACT_HASHES.json` with 0 mismatches.
