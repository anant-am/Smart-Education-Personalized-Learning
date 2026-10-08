# Academic Evaluation Report — Model 4: Autoencoder + Recommender Network
**Architecture:** Non-Linear Bottleneck Autoencoder (Input 500 -> Latent 32 -> Reconstructed 500) + Multi-Resource Scoring Head  
**Parameter Count:** 103,368  
**Evaluation Date:** 2026-09-12 15:31:15  
**Device / Hardware:** cuda (NVIDIA RTX 3050 Laptop GPU 6GB VRAM)

## 1. SMOTE Oversampling & Isolation Verification
- **Target Partition:** Development Training Set (`train.csv`) Exclusively  
- **Pre-SMOTE Counts:** Class 0 (Incorrect) = 141,257, Class 1 (Correct) = 284,418  
- **Post-SMOTE Counts:** Class 0 = 284,418, Class 1 = 284,418 (Balanced 1:1)  
- **Validation & Test Sets:** 100% untouched in their natural class distribution (zero synthetic records).

## 2. 10-Fold Cross-Validation Summary (Development Interactions)
- **Mean Reconstruction MSE:** 0.1502 ± 0.0095

## 3. Initial Training & Overfitting Diagnosis
- **Diagnosis:** Healthy Fit / Well-Regularized
- **Recent Train Loss:** 0.1212 | **Recent Val Loss:** 0.0901
- **Loss Gap:** -0.0311
- **Applied Correction:** Fine-tuning with conservative learning rate (5e-4)

## 4. Recommendation Evaluation on Isolated 250 Unseen Test Students
| Metric | Value | Interpretation |
|---|---|---|
| **Reconstruction MSE** | **0.1104** | Interaction reconstruction fidelity |
| **Precision@5** | 0.1618 | Relevant resources among top 5 |
| **Recall@5** | 0.0081 | Coverage of student desired interactions |
| **NDCG@5** | **0.1757** | Normalized discounted cumulative gain |
| **HitRate@5** | **48.55%** | Percentage of students with ≥1 relevant item in top 5 |
| **Precision@10** | 0.1647 | Relevant resources among top 10 |
| **Recall@10** | 0.0207 | Coverage of student desired interactions |
| **NDCG@10** | **0.1731** | Normalized discounted cumulative gain |
| **HitRate@10** | **65.56%** | Percentage of students with ≥1 relevant item in top 10 |

## 4. Methodological Safeguards Verified
- **Resource Vocabulary Isolation:** Active learning resources fitted strictly from Development interactions; 250 test students had zero influence on resource selection.
- **Zero Cohort Overlap:** Verified $\text{Dev} \cap \text{Test} = \emptyset$.
- **Retraining Invariant:** Retrained model instantiated as a fresh `AutoencoderRecommender` instance with a separate `ModelTrainer` and fresh `AdamW` optimizer.