# Advanced Data Augmentation & Resampling Comparative Benchmark
**Project:** AI-Based Personalized Learning Recommendation System – Smart Education  
**Evaluation Cohort:** `KT-3 (250) TEST` (249 Valid Students, 174,608 Events strictly untouched)  
**Date:** September 12, 2026

## 1. Executive Summary & Defense Findings
- **Class Imbalance Context:** Real-world educational logs suffer from acute positive skew (66.8% correct vs 33.2% incorrect). Left uncorrected, standard models favor majority-class predictions and fail to detect struggling student failure states.
- **Four Evaluated Paradigms:**
  1. **Baseline (Natural):** Unmodified imbalanced training distribution ($0.497 : 1$).
  2. **SMOTE:** Uniform linear interpolation along nearest-neighbor line segments ($1.00 : 1$).
  3. **ADASYN:** Adaptive density synthesis concentrating on hard-to-learn decision boundary instances ($1.018 : 1$).
  4. **Tabular WGAN-GP:** Conditional Generative Adversarial Network learning joint continuous manifold distributions ($1.00 : 1$).
- **Key Finding:** All three rebalancing strategies dramatically increase **Minority Recall (Class 0: Incorrect)** on unseen students compared to the imbalanced baseline, enabling the system to reliably catch student deficiencies for remedial intervention.

## 2. Quantitative Benchmark Table on Unseen Test Cohort

| Technique | Train Samples | 0:1 Ratio | Test Acc | Test ROC-AUC | Test PR-AUC | Minority Recall (0) | Minority Prec (0) | Minority F1 (0) | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Natural)** | 425,675 | 0.497 | 0.6708 | 0.6052 | 0.7518 | **0.0232** | 0.4840 | **0.0442** | 0.4227 |
| **SMOTE** | 568,836 | 1.000 | 0.5846 | 0.6098 | 0.7553 | **0.5536** | 0.4038 | **0.4670** | 0.5633 |
| **ADASYN** | 574,080 | 1.018 | 0.5830 | 0.6066 | 0.7492 | **0.5582** | 0.4031 | **0.4681** | 0.5626 |
| **Tabular WGAN-GP** | 568,836 | 1.000 | 0.5902 | 0.5847 | 0.7318 | **0.4625** | 0.3948 | **0.4260** | 0.5537 |

## 3. Methodological Comparison & Theoretical Tradeoffs
| Attribute | Standard SMOTE | ADASYN | Tabular WGAN-GP |
|---|---|---|---|
| **Synthesis Mechanism** | Uniform linear interpolation between $k$-NN | Weighted density interpolation $\Gamma_i$ | Non-linear neural generator $G(z, c)$ |
| **Boundary Sensitivity** | Homogeneous across all minority points | Concentrates on ambiguous boundary regions | Continuous global manifold modeling |
| **Susceptibility to Noise** | Can bridge outliers into majority space | May over-amplify noise if outliers have high $r_i$ | Gradient penalty Lipschitz bound resists outlier drift |
| **Compute Complexity** | Minimal ($O(N \log N)$ with KD-Tree) | Fast ($O(N \log N)$ density estimation) | High (Iterative minimax neural training) |
| **Best Educational Use Case** | Fast, stable baseline class balancing | Pinpointing borderline students on difficult concepts | Simulating diverse student behavior archetypes |

## 4. Academic Integrity & Non-Leakage Proof
- **Training-Only Isolation:** SMOTE, ADASYN, and WGAN-GP were strictly fitted on `train.csv`. The validation set (`val.csv`) and unseen test cohort (`test.csv`) remained 100% untouched.
- **Chronological Sequence Integrity:** Sequence KT models continue to receive authentic timestamped interactions; resampling is targeted at tabular feature representations and diagnostic decision boundaries.