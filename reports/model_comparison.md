# Five Hybrid Deep-Learning Models — Comprehensive Academic Comparison

**Project:** AI-Based Personalized Learning Recommendation System – Smart Education  
**Dataset:** EdNet-KT3 (1,000-User Development Cohort, 250-User Unseen Test Cohort)  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM), CUDA 12.6, PyTorch  

## 1. Master Cross-Model Performance Evaluation Table

| Model ID | Hybrid Architecture | Primary Task | Parameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | LogLoss | Training Time (s) | Inference Time (s) | Peak GPU (MB) | NDCG@5 | HitRate@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model 1 | LSTM + Attention | Knowledge Tracing | 803,201 | 0.6584 | 0.6689 | 0.9542 | 0.7865 | 0.5760 | 0.7078 | 0.6340 | 2.21 | 0.0485 | 58.2 | N/A | N/A |
| Model 2 | Transformer + Knowledge Tracing | Knowledge Tracing | 824,385 | 0.6587 | 0.6607 | 0.9915 | 0.7930 | 0.5786 | 0.7095 | 0.6340 | 3.96 | 0.0990 | 58.8 | N/A | N/A |
| Model 3 | BERT-style Transformer + NCF | Collaborative Tracing | 1,572,737 | 0.6594 | 0.6598 | 0.9978 | 0.7943 | 0.5442 | 0.6921 | 0.6403 | 2.79 | 0.0755 | 68.8 | N/A | N/A |
| Model 4 | Autoencoder + Recommender Network | Resource Recommendation | 103,368 | MSE: 0.1095 | 0.1809 (@5) | 0.0084 (@5) | N/A | N/A | N/A | N/A | 2.74 | N/A | 20.6 | 0.1571 | 47.30% |
| Model 5 | CNN + LSTM | Temporal Knowledge Tracing | 803,201 | 0.9884 | 0.9871 | 0.9954 | 0.9912 | 0.9996 | 0.9998 | 0.0259 | 2.91 | 0.0517 | 68.8 | N/A | N/A |


## 2. Cross-Validation Summary (10-Fold Grouped on Development Cohort)

| Model | CV Accuracy (Mean ± Std) | CV F1-Score (Mean ± Std) | CV ROC-AUC (Mean ± Std) | CV Log Loss (Mean ± Std) |
|---|---|---|---|---|
| **Model 1 (LSTM + Attention)** | 0.6443 ± 0.0126 | 0.7818 ± 0.0104 | 0.5679 ± 0.0144 | 0.6449 ± 0.0072 |
| **Model 2 (Transformer + KT)** | 0.6440 ± 0.0131 | 0.7834 ± 0.0096 | 0.5559 ± 0.0126 | 0.6482 ± 0.0083 |
| **Model 3 (BERT-style Transformer + NCF)** | 0.6431 ± 0.0135 | 0.7821 ± 0.0099 | 0.5194 ± 0.0131 | 0.6534 ± 0.0064 |
| **Model 4 (Autoencoder Recommender)** | N/A (Reconstruction MSE) | N/A | CV MSE: 0.1467 ± 0.0079 | N/A |
| **Model 5 (CNN + LSTM)** | 0.6440 ± 0.0131 | 0.7834 ± 0.0096 | 0.5915 ± 0.0160 | 0.6441 ± 0.0102 |


## 3. Academic Analysis & Architectural Strengths

1. **Model 1 (LSTM + Attention):** Combines recurrence with self-attention weights to capture temporal dependencies with explainable attention maps.
2. **Model 2 (Transformer + Knowledge Tracing):** Uses causal multi-head self-attention to model long-range dependencies with O(1) sequential path length.
3. **Model 3 (BERT-style Transformer + NCF):** Combines bidirectional sequence representations with non-linear collaborative filtering item interactions.
4. **Model 4 (Autoencoder + Recommender Network):** Compresses high-dimensional engagement into a latent bottleneck, reconstructing profiles and ranking authentic EdNet interventions.
5. **Model 5 (CNN + LSTM):** Extracts localized temporal features via 1D convolutions before tracking cumulative learning trajectories with LSTM sequence cells.