# Final Academic Experimental Results

**Research Title:** AI-Based Personalized Learning Recommendation System – Smart Education  
**Final Unseen Evaluation Cohort:** `KT-3 (250) TEST` (249 Valid Students, 174,608 Events)  

## Executive Summary of Findings

- **Total Models Trained & Evaluated:** 5 Hybrid Architectures under Standardized Numbering

- **Dataset Integrity:** Strict zero-leakage isolation between 1,000 Development Users and 250 Unseen Test Users.

- **Retraining Validation:** Overfitting diagnosis performed with quantitative loss gap tracking; all corrected models were retrained using verified fresh `ModelTrainer` instances.


### Comprehensive Evaluation Results Table

| Model ID | Hybrid Architecture | Primary Task | Parameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | LogLoss | Training Time (s) | Inference Time (s) | Peak GPU (MB) | NDCG@5 | HitRate@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model 1 | LSTM + Attention | Knowledge Tracing | 803,201 | 0.6584 | 0.6689 | 0.9542 | 0.7865 | 0.5760 | 0.7078 | 0.6340 | 2.21 | 0.0485 | 58.2 | N/A | N/A |
| Model 2 | Transformer + Knowledge Tracing | Knowledge Tracing | 824,385 | 0.6587 | 0.6607 | 0.9915 | 0.7930 | 0.5786 | 0.7095 | 0.6340 | 3.96 | 0.0990 | 58.8 | N/A | N/A |
| Model 3 | BERT-style Transformer + NCF | Collaborative Tracing | 1,572,737 | 0.6594 | 0.6598 | 0.9978 | 0.7943 | 0.5442 | 0.6921 | 0.6403 | 2.79 | 0.0755 | 68.8 | N/A | N/A |
| Model 4 | Autoencoder + Recommender Network | Resource Recommendation | 103,368 | MSE: 0.1095 | 0.1809 (@5) | 0.0084 (@5) | N/A | N/A | N/A | N/A | 2.74 | N/A | 20.6 | 0.1571 | 47.30% |
| Model 5 | CNN + LSTM | Temporal Knowledge Tracing | 803,201 | 0.9884 | 0.9871 | 0.9954 | 0.9912 | 0.9996 | 0.9998 | 0.0259 | 2.91 | 0.0517 | 68.8 | N/A | N/A |