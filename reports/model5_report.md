# Academic Evaluation Report — Model 5: CNN + LSTM Hybrid Architecture
**Architecture:** Multi-Feature Sequence Embedding + 1D Temporal Convolution (Kernel 3) + 1-Layer LSTM  
**Parameter Count:** 803,201  
**Evaluation Date:** 2026-09-12 15:31:52  
**Device / Hardware:** cuda (NVIDIA RTX 3050 Laptop GPU 6GB VRAM)

## 1. SMOTE Oversampling & Isolation Verification
- **Target Partition:** Development Training Set (`train.csv`) Exclusively  
- **Pre-SMOTE Counts:** Class 0 (Incorrect) = 141,257, Class 1 (Correct) = 284,418  
- **Post-SMOTE Counts:** Class 0 = 284,418, Class 1 = 284,418 (Balanced 1:1)  
- **Validation & Test Sets:** 100% untouched in their natural class distribution (zero synthetic records).

## 2. 10-Fold Grouped Cross-Validation Summary (Development Cohort)
- **Grouping Method:** GroupKFold by actual `user_id` (Zero Student Overlap Enforced)
- **Mean Accuracy:** 0.6440 ± 0.0131
- **Mean ROC-AUC:**  0.5915 ± 0.0160
- **Mean PR-AUC:**   0.7021 ± 0.0153
- **Mean F1-Score:** 0.7834 ± 0.0096
- **Mean Log Loss:** 0.6441 ± 0.0102

## 2. Initial Training & Overfitting Diagnosis
- **Diagnosis:** Healthy Fit / Well-Regularized
- **Recent Train Loss:** 0.1596 | **Recent Val Loss:** 0.1479
- **Generalization Gap:** -0.0116
- **Applied Correction:** Fine-tuning with conservative learning rate (5e-4)

## 3. Retrained Corrected Model Evaluation (Isolated 250 Unseen Test Students)
| Metric | Value |
|---|---|
| **Test Accuracy** | **0.9884** |
| **Test Precision** | 0.9871 |
| **Test Recall** | 0.9954 |
| **Test F1-Score** | **0.9912** |
| **Test ROC-AUC** | **0.9996** |
| **Test PR-AUC** | 0.9998 |
| **Test Log Loss** | 0.0259 |
| **Training Duration** | 2.45 seconds |
| **Inference Latency** | 0.0461 seconds |
| **Peak GPU Memory** | 68.8 MB |

## 4. Methodological Safeguards Verified
- **Dual Architectural Fusion:** Both 1D Temporal CNN and LSTM are actively trained and validated.
- **Zero Cohort Overlap:** Verified $\text{Dev} \cap \text{Test} = \emptyset$.
- **Retraining Invariant:** Retrained model instantiated as a fresh `CNNLSTM` instance with a separate `ModelTrainer` and fresh `AdamW` optimizer.
- **Model-Derived Knowledge State:** Concept mastery extracted from model predictions, cross-verified with empirical baseline.