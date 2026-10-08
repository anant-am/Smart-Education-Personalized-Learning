# Academic Evaluation Report — Model 2: Transformer + Knowledge Tracing
**Architecture:** Multi-Feature Embedding + Positional Encoding + 2-Layer Causal Transformer Encoder (4 Heads, Dim 64)  
**Parameter Count:** 824,385  
**Evaluation Date:** 2026-09-12 15:30:13  
**Device / Hardware:** cuda (NVIDIA RTX 3050 Laptop GPU 6GB VRAM)

## 1. SMOTE Oversampling & Isolation Verification
- **Target Partition:** Development Training Set (`train.csv`) Exclusively  
- **Pre-SMOTE Counts:** Class 0 (Incorrect) = 141,257, Class 1 (Correct) = 284,418  
- **Post-SMOTE Counts:** Class 0 = 284,418, Class 1 = 284,418 (Balanced 1:1)  
- **Validation & Test Sets:** 100% untouched in their natural class distribution (zero synthetic records).

## 2. 10-Fold Grouped Cross-Validation Summary (Development Cohort)
- **Grouping Method:** GroupKFold by actual `user_id` (Zero Student Overlap Enforced)
- **Mean Accuracy:** 0.6440 ± 0.0131
- **Mean ROC-AUC:**  0.5559 ± 0.0126
- **Mean PR-AUC:**   0.6806 ± 0.0184
- **Mean F1-Score:** 0.7834 ± 0.0096
- **Mean Log Loss:** 0.6482 ± 0.0083

## 2. Initial Training & Overfitting Diagnosis
- **Diagnosis:** Healthy Fit / Well-Regularized
- **Recent Train Loss:** 0.6477 | **Recent Val Loss:** 0.6434
- **Generalization Gap:** -0.0042
- **Applied Correction:** Fine-tuning with conservative learning rate (5e-4)

## 3. Retrained Corrected Model Evaluation (Isolated 250 Unseen Test Students)
| Metric | Value |
|---|---|
| **Test Accuracy** | **0.6587** |
| **Test Precision** | 0.6607 |
| **Test Recall** | 0.9915 |
| **Test F1-Score** | **0.7930** |
| **Test ROC-AUC** | **0.5786** |
| **Test PR-AUC** | 0.7095 |
| **Test Log Loss** | 0.6340 |
| **Training Duration** | 3.47 seconds |
| **Inference Latency** | 0.0544 seconds |
| **Peak GPU Memory** | 58.8 MB |

## 4. Methodological Safeguards Verified
- **Strict Causal Masking:** Upper-triangular self-attention masking prevents attention to future interactions.
- **Zero Cohort Overlap:** Verified $\text{Dev} \cap \text{Test} = \emptyset$.
- **Retraining Invariant:** Retrained model instantiated as a fresh `TransformerKT` instance with a separate `ModelTrainer` and fresh `AdamW` optimizer.
- **Model-Derived Knowledge State:** Concept mastery extracted from model predictions, cross-verified with empirical baseline.