# Adaptive Synthetic Sampling (ADASYN) Report
**Target Partition:** Development Training Set (`train.csv`) Exclusively  
**Validation Partition (`val.csv`):** 100% UNTOUCHED (Natural Distribution Preserved)  
**Final Test Partition (`test.csv`):** 100% UNTOUCHED (Strictly Isolated)

## 1. Quantitative Class Rebalancing Summary
| Split / Status | Class 0 (Incorrect) | Class 1 (Correct) | Total Samples | Imbalance Ratio (0:1) |
|---|---|---|---|---|
| **Before ADASYN (train.csv)** | 124,998 (31.5%) | 271,292 (68.5%) | 396,290 | 0.461 : 1 |
| **After ADASYN (train_adasyn_flat.csv)** | 258,614 (48.8%) | 271,292 (51.2%) | 529,906 | 0.953 : 1 |

## 2. Theoretical Mechanism & Academic Justification
- **Adaptive Density Distribution:** Unlike standard SMOTE, which generates synthetic minority examples uniformly along $k$-NN segments, ADASYN uses a density distribution $\Gamma_i = r_i / \sum r_i$ where $r_i = \Delta_i / K$ (the proportion of majority-class examples among $K$ nearest neighbors of minority instance $x_i$).
- **Focus on Boundary Hardness:** More synthetic samples are adaptively generated for minority instances that are harder to learn (those near the ambiguous boundary between correct and incorrect student actions).
- **Zero Distribution Leakage:** ADASYN was fitted and executed exclusively on `train.csv`. Validation (`val.csv`) and Unseen Test (`test.csv`) sets remain in their untouched natural distribution.
- **Algorithm Hyperparameters:** $K = 5$ nearest neighbors (`RandomState=42`).
- **Feature Space (10 variables):** `previous_accuracy, recent_accuracy_5, attempt_count, response_time_norm, time_since_prev_norm, source_encoded, platform_encoded, part, num_responses, question_idx`.