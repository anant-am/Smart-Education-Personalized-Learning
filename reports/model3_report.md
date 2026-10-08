# Academic Evaluation Report — Model 3: BERT-style Transformer + Neural Collaborative Filtering
**Architecture:** Multi-Feature Sequence Embedding + BERT-style Transformer Encoder + NCF (GMF & MLP) NeuMF Fusion Head  
**Parameter Count:** 1,572,737  
**Evaluation Date:** 2026-09-12 15:30:53  
**Device / Hardware:** cuda (NVIDIA RTX 3050 Laptop GPU 6GB VRAM)

## 1. SMOTE Oversampling & Isolation Verification
- **Target Partition:** Development Training Set (`train.csv`) Exclusively  
- **Pre-SMOTE Counts:** Class 0 (Incorrect) = 141,257, Class 1 (Correct) = 284,418  
- **Post-SMOTE Counts:** Class 0 = 284,418, Class 1 = 284,418 (Balanced 1:1)  
- **Validation & Test Sets:** 100% untouched in their natural class distribution (zero synthetic records).

## 2. 10-Fold Grouped Cross-Validation Summary (Development Cohort)
- **Grouping Method:** GroupKFold by actual `user_id` (Zero Student Overlap Enforced)
- **Mean Accuracy:** 0.6431 ± 0.0135
- **Mean ROC-AUC:**  0.5194 ± 0.0131
- **Mean PR-AUC:**   0.6579 ± 0.0141
- **Mean F1-Score:** 0.7821 ± 0.0099
- **Mean Log Loss:** 0.6534 ± 0.0064

## 2. Initial Training & Overfitting Diagnosis
- **Diagnosis:** Healthy Fit / Well-Regularized
- **Recent Train Loss:** 0.6488 | **Recent Val Loss:** 0.6492
- **Generalization Gap:** 0.0004
- **Applied Correction:** Fine-tuning with conservative learning rate (5e-4)

## 3. Retrained Corrected Model Evaluation (Isolated 250 Unseen Test Students)
| Metric | Value |
|---|---|
| **Test Accuracy** | **0.6594** |
| **Test Precision** | 0.6598 |
| **Test Recall** | 0.9978 |
| **Test F1-Score** | **0.7943** |
| **Test ROC-AUC** | **0.5442** |
| **Test PR-AUC** | 0.6921 |
| **Test Log Loss** | 0.6403 |
| **Training Duration** | 2.44 seconds |
| **Inference Latency** | 0.0538 seconds |
| **Peak GPU Memory** | 68.8 MB |

## 4. Methodological Safeguards Verified
- **Explicit Hybrid Attribution:** Naming correctly documented as BERT-style Transformer + NCF without misleading claims of natural-language text pretraining.
- **Zero Cohort Overlap:** Verified $\text{Dev} \cap \text{Test} = \emptyset$.
- **Retraining Invariant:** Retrained model instantiated as a fresh `BERTNCF` instance with a separate `ModelTrainer` and fresh `AdamW` optimizer.
- **Model-Derived Knowledge State:** Concept mastery extracted from model predictions, cross-verified with empirical baseline.