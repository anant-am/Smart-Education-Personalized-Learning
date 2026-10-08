# Synthetic Minority Over-sampling Technique (SMOTE) Report
**Target Partition:** Development Training Set (`train.csv`) Exclusively  
**Validation Partition (`val.csv`):** 100% UNTOUCHED (Natural Distribution Preserved)  
**Final Test Partition (`test.csv`):** 100% UNTOUCHED (Strictly Isolated)

## 1. Quantitative Class Rebalancing Summary
| Split / Status | Class 0 (Incorrect) | Class 1 (Correct) | Total Samples | Imbalance Ratio (0:1) |
|---|---|---|---|---|
| **Before SMOTE (train.csv)** | 141,257 (33.2%) | 284,418 (66.8%) | 425,675 | 0.497 : 1 |
| **After SMOTE (train_smote_flat.csv)** | 284,418 (50.0%) | 284,418 (50.0%) | 568,836 | 1.00 : 1 |

## 2. Academic Defense & Methodological Guarantees
- **Zero Distribution Leakage:** SMOTE synthetic generation was fitted and executed exclusively on the training partition. The validation fold and the 250-student final test cohort remain in their natural un-augmented distribution, preventing synthetic data leakage into evaluation benchmarks.
- **Sequence Causal Integrity:** Sequence models (LSTM, Transformer, BERT-NCF, CNN-LSTM) are trained on authentic chronological student interaction streams; SMOTE is applied to derived tabular decision-boundary representations to prevent temporal graph disruption.
- **Algorithm Configuration:** Regular SMOTE with $k=5$ nearest neighbors in feature space (`RandomState=42`).
- **Feature Space (10 variables):** `previous_accuracy, recent_accuracy_5, attempt_count, response_time_norm, time_since_prev_norm, source_encoded, platform_encoded, part, num_responses, question_idx`.