# Academic Evaluation Report — Model 1: LSTM + Attention (Knowledge Tracing)
**Architecture:** Multi-Feature Embedding (Item + Concept + Part) + 1-Layer LSTM + Causal Scaled Dot-Product Self-Attention  
**Parameter Count:** 803,201  
**Evaluation Date:** 2026-09-12 15:28:19  
**Device / Hardware:** cuda (NVIDIA RTX 3050 Laptop GPU 6GB VRAM)

## 1. SMOTE Oversampling & Isolation Verification
- **Target Partition:** Development Training Set (`train.csv`) Exclusively  
- **Pre-SMOTE Counts:** Class 0 (Incorrect) = 141,257, Class 1 (Correct) = 284,418  
- **Post-SMOTE Counts:** Class 0 = 284,418, Class 1 = 284,418 (Balanced 1:1)  
- **Validation & Test Sets:** 100% untouched in their natural class distribution (zero synthetic records).

## 2. 10-Fold Grouped Cross-Validation Summary (Development Cohort)
- **Grouping Method:** GroupKFold by actual `user_id` (Zero Student Overlap Enforced)
- **Mean Accuracy:** 0.6440 ± 0.0130
- **Mean ROC-AUC:**  0.5687 ± 0.0083
- **Mean PR-AUC:**   0.6876 ± 0.0127
- **Mean F1-Score:** 0.7830 ± 0.0095
- **Mean Log Loss:** 0.6452 ± 0.0083

## 2. Initial Training & Overfitting Diagnosis
- **Diagnosis:** Healthy Fit / Well-Regularized
- **Recent Train Loss:** 0.6226 | **Recent Val Loss:** 0.6521
- **Generalization Gap:** 0.0295
- **Applied Correction:** Fine-tuning with conservative learning rate (5e-4)

## 3. Retrained Corrected Model Evaluation (Isolated 250 Unseen Test Students)
| Metric | Value |
|---|---|
| **Test Accuracy** | **0.6628** |
| **Test Precision** | 0.6713 |
| **Test Recall** | 0.9569 |
| **Test F1-Score** | **0.7891** |
| **Test ROC-AUC** | **0.5811** |
| **Test PR-AUC** | 0.7121 |
| **Test Log Loss** | 0.6319 |
| **Training Duration** | 6.90 seconds |
| **Inference Latency** | 0.2097 seconds |
| **Peak GPU Memory** | 58.8 MB |

## 4. Methodological Safeguards Verified
- **Zero Target Leakage:** Target outcome `is_correct` excluded from input features.
- **Zero Cohort Overlap:** Verified $\text{Dev} \cap \text{Test} = \emptyset$.
- **Retraining Invariant:** Retrained model instantiated as a fresh `LSTMAttentionKT` instance with a separate `ModelTrainer` and fresh `AdamW` optimizer.
- **Model-Derived Knowledge State:** Concept mastery extracted from model predictions, cross-verified with empirical baseline.