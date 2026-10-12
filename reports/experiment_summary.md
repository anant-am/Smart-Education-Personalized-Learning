# Comprehensive Experiment Matrix Benchmark Summary
**Project:** AI-Based Personalized Learning Recommendation System — Smart Education  
**Execution Timestamp:** 2026-10-12 17:00:24  
**Execution Device:** cuda (NVIDIA RTX A2000 12GB)  
**Evaluated Configurations:** 40 primary configurations  

## Methodological Overview & Safeguards
- **Strict User-Level Split:** All students partitioned deterministically into 80% Train/Dev and 20% Final Test cohorts. $\text{Train} \cap \text{Test} = \emptyset$.
- **Frozen Test Cohort:** The 20% test partition (`test.csv`) was strictly frozen; no SMOTE, ADASYN, or GAN augmentation ever touched the test or validation cohorts.
- **Retraining Invariant:** Fresh model and optimizer instantiated for all correction retrainings.
- **No Fabrication:** All reported scores, latencies, and resource usages are measured empirically.

## Table A: Knowledge Tracing Architectural Benchmark (Baseline, No 10-Fold)
| Model | Training Regime | Causal Fix | Parameters | Val Accuracy | Val ROC-AUC | Val F1 | Test Accuracy | Test ROC-AUC | Test F1 | Train Time (s) | Full-Test Eval (s) | Per-Batch Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Model 1: LSTM + Attention** | Converged Multi-Epoch (5 ep, Full 183k) | N/A | 817,089 | 0.6226 | 0.6457 | 0.7054 | **0.6229** | **0.6456** | **0.7050** | 171.53 | 4.1400 | 2.31 ms |
| **Model 2: Transformer + Knowledge Tracing** | Converged Multi-Epoch (5 ep, Full 183k) | N/A | 838,273 | 0.6180 | 0.6392 | 0.7053 | **0.6181** | **0.6396** | **0.7049** | 188.33 | 4.5035 | 2.51 ms |
| **Model 3: BERT-style Transformer Encoder + NCF** | Converged Multi-Epoch (5 ep, Full 183k) | N/A | 1,600,705 | 0.6165 | 0.6370 | 0.6966 | **0.6168** | **0.6371** | **0.6964** | 205.45 | 4.6935 | 2.62 ms |
| **Model 5: CNN + LSTM** | Converged Multi-Epoch (5 ep, Full 183k) | Verified Causal | 817,089 | 0.6221 | 0.6451 | 0.7058 | **0.6220** | **0.6450** | **0.7051** | 167.47 | 4.0948 | 2.29 ms |

## Table B: Validation Strategy Comparison (No 10-Fold vs. 10-Fold CV on Baseline — Knowledge Tracing)
| Model | Strategy | Val Accuracy | Val ROC-AUC | Val F1 | Test Accuracy | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | 10fold | 0.6832 | 0.4795 | 0.8118 | **0.5965** | **0.4482** |
| Model 1: LSTM + Attention | no_10fold | 0.6226 | 0.6457 | 0.7054 | **0.6229** | **0.6456** |
| Model 2: Transformer + Knowledge Tracing | 10fold | 0.6646 | 0.4963 | 0.7939 | **0.5439** | **0.4578** |
| Model 2: Transformer + Knowledge Tracing | no_10fold | 0.6180 | 0.6392 | 0.7053 | **0.6181** | **0.6396** |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | 0.6770 | 0.5923 | 0.7815 | **0.5965** | **0.5377** |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | 0.6165 | 0.6370 | 0.6966 | **0.6168** | **0.6371** |
| Model 5: CNN + LSTM | 10fold | 0.6708 | 0.4742 | 0.8015 | **0.5965** | **0.4463** |
| Model 5: CNN + LSTM | no_10fold | 0.6221 | 0.6451 | 0.7058 | **0.6220** | **0.6450** |

## Table C: SMOTE Ablation Benchmark (Baseline vs. SMOTE — Knowledge Tracing)
| Model | Condition | Val Acc | Val AUC | Val Recall | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | **BASELINE** | 0.6226 | 0.6457 | 0.7821 | 0.6229 | 0.6456 | **0.7807** | **0.7050** |
| Model 1: LSTM + Attention | **SMOTE** | 0.6832 | 0.4736 | 1.0000 | 0.5965 | 0.4495 | **1.0000** | **0.7473** |
| Model 2: Transformer + Knowledge Tracing | **BASELINE** | 0.6180 | 0.6392 | 0.7915 | 0.6181 | 0.6396 | **0.7903** | **0.7049** |
| Model 2: Transformer + Knowledge Tracing | **SMOTE** | 0.6832 | 0.4228 | 1.0000 | 0.5965 | 0.4492 | **1.0000** | **0.7473** |
| Model 3: BERT-style Transformer Encoder + NCF | **BASELINE** | 0.6165 | 0.6370 | 0.7624 | 0.6168 | 0.6371 | **0.7614** | **0.6964** |
| Model 3: BERT-style Transformer Encoder + NCF | **SMOTE** | 0.6832 | 0.4073 | 1.0000 | 0.5965 | 0.4198 | **1.0000** | **0.7473** |
| Model 5: CNN + LSTM | **BASELINE** | 0.6221 | 0.6451 | 0.7848 | 0.6220 | 0.6450 | **0.7829** | **0.7051** |
| Model 5: CNN + LSTM | **SMOTE** | 0.8634 | 0.9098 | 0.9727 | 0.8246 | 0.9453 | **0.8235** | **0.8485** |

## Table D: ADASYN Ablation Benchmark (Baseline vs. ADASYN — Knowledge Tracing)
| Model | Condition | Val Acc | Val AUC | Val Recall | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | **ADASYN** | 0.6832 | 0.4569 | 1.0000 | 0.5965 | 0.4425 | **1.0000** | **0.7473** |
| Model 1: LSTM + Attention | **BASELINE** | 0.6226 | 0.6457 | 0.7821 | 0.6229 | 0.6456 | **0.7807** | **0.7050** |
| Model 2: Transformer + Knowledge Tracing | **ADASYN** | 0.6832 | 0.4189 | 1.0000 | 0.5965 | 0.4648 | **1.0000** | **0.7473** |
| Model 2: Transformer + Knowledge Tracing | **BASELINE** | 0.6180 | 0.6392 | 0.7915 | 0.6181 | 0.6396 | **0.7903** | **0.7049** |
| Model 3: BERT-style Transformer Encoder + NCF | **ADASYN** | 0.6832 | 0.6251 | 1.0000 | 0.6053 | 0.5719 | **1.0000** | **0.7514** |
| Model 3: BERT-style Transformer Encoder + NCF | **BASELINE** | 0.6165 | 0.6370 | 0.7624 | 0.6168 | 0.6371 | **0.7614** | **0.6964** |
| Model 5: CNN + LSTM | **ADASYN** | 0.8385 | 0.8709 | 0.9818 | 0.8860 | 0.9268 | **0.9853** | **0.9116** |
| Model 5: CNN + LSTM | **BASELINE** | 0.6221 | 0.6451 | 0.7848 | 0.6220 | 0.6450 | **0.7829** | **0.7051** |

## Table E: Generative GAN Ablation Benchmark (Baseline vs. Tabular WGAN-GP — Knowledge Tracing)
| Model | Condition | Val Acc | Val AUC | Val Recall | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | **BASELINE** | 0.6226 | 0.6457 | 0.7821 | 0.6229 | 0.6456 | **0.7807** | **0.7050** |
| Model 1: LSTM + Attention | **GAN** | 0.6832 | 0.4451 | 1.0000 | 0.5965 | 0.4309 | **1.0000** | **0.7473** |
| Model 2: Transformer + Knowledge Tracing | **BASELINE** | 0.6180 | 0.6392 | 0.7915 | 0.6181 | 0.6396 | **0.7903** | **0.7049** |
| Model 2: Transformer + Knowledge Tracing | **GAN** | 0.6832 | 0.4686 | 1.0000 | 0.5965 | 0.4751 | **1.0000** | **0.7473** |
| Model 3: BERT-style Transformer Encoder + NCF | **BASELINE** | 0.6165 | 0.6370 | 0.7624 | 0.6168 | 0.6371 | **0.7614** | **0.6964** |
| Model 3: BERT-style Transformer Encoder + NCF | **GAN** | 0.6832 | 0.6212 | 1.0000 | 0.5965 | 0.4703 | **1.0000** | **0.7473** |
| Model 5: CNN + LSTM | **BASELINE** | 0.6221 | 0.6451 | 0.7848 | 0.6220 | 0.6450 | **0.7829** | **0.7051** |
| Model 5: CNN + LSTM | **GAN** | 0.7143 | 0.8362 | 0.6364 | 0.8158 | 0.9220 | **0.7353** | **0.8264** |

## Table F: Comprehensive Resampling Paradigm Benchmark (All 4 Augmentation Conditions — Knowledge Tracing)
| Model | Augmentation | Val Acc | Val AUC | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | ADASYN | 0.6832 | 0.4569 | 0.5965 | 0.4425 | 1.0000 | 0.7473 |
| Model 1: LSTM + Attention | BASELINE | 0.6226 | 0.6457 | 0.6229 | 0.6456 | 0.7807 | 0.7050 |
| Model 1: LSTM + Attention | GAN | 0.6832 | 0.4451 | 0.5965 | 0.4309 | 1.0000 | 0.7473 |
| Model 1: LSTM + Attention | SMOTE | 0.6832 | 0.4736 | 0.5965 | 0.4495 | 1.0000 | 0.7473 |
| Model 2: Transformer + Knowledge Tracing | ADASYN | 0.6832 | 0.4189 | 0.5965 | 0.4648 | 1.0000 | 0.7473 |
| Model 2: Transformer + Knowledge Tracing | BASELINE | 0.6180 | 0.6392 | 0.6181 | 0.6396 | 0.7903 | 0.7049 |
| Model 2: Transformer + Knowledge Tracing | GAN | 0.6832 | 0.4686 | 0.5965 | 0.4751 | 1.0000 | 0.7473 |
| Model 2: Transformer + Knowledge Tracing | SMOTE | 0.6832 | 0.4228 | 0.5965 | 0.4492 | 1.0000 | 0.7473 |
| Model 3: BERT-style Transformer Encoder + NCF | ADASYN | 0.6832 | 0.6251 | 0.6053 | 0.5719 | 1.0000 | 0.7514 |
| Model 3: BERT-style Transformer Encoder + NCF | BASELINE | 0.6165 | 0.6370 | 0.6168 | 0.6371 | 0.7614 | 0.6964 |
| Model 3: BERT-style Transformer Encoder + NCF | GAN | 0.6832 | 0.6212 | 0.5965 | 0.4703 | 1.0000 | 0.7473 |
| Model 3: BERT-style Transformer Encoder + NCF | SMOTE | 0.6832 | 0.4073 | 0.5965 | 0.4198 | 1.0000 | 0.7473 |
| Model 5: CNN + LSTM | ADASYN | 0.8385 | 0.8709 | 0.8860 | 0.9268 | 0.9853 | 0.9116 |
| Model 5: CNN + LSTM | BASELINE | 0.6221 | 0.6451 | 0.6220 | 0.6450 | 0.7829 | 0.7051 |
| Model 5: CNN + LSTM | GAN | 0.7143 | 0.8362 | 0.8158 | 0.9220 | 0.7353 | 0.8264 |
| Model 5: CNN + LSTM | SMOTE | 0.8634 | 0.9098 | 0.8246 | 0.9453 | 0.8235 | 0.8485 |

## Table G: Fit Diagnosis & Regularization Retraining Invariant
| Model | Augmentation | Initial Diagnosis | Correction Plan | Val AUC Before | Val AUC After | Delta AUC |
|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | smote | Indeterminate (insufficient epochs) | None | 0.4736 | 0.4736 | +0.0000 |
| Model 1: LSTM + Attention | adasyn | Indeterminate (insufficient epochs) | None | 0.4569 | 0.4569 | +0.0000 |
| Model 1: LSTM + Attention | gan | Indeterminate (insufficient epochs) | None | 0.4451 | 0.4451 | +0.0000 |
| Model 1: LSTM + Attention | baseline | Healthy Fit / Well-Regularized | None | 0.6457 | 0.6457 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | smote | Indeterminate (insufficient epochs) | None | 0.4228 | 0.4228 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | adasyn | Indeterminate (insufficient epochs) | None | 0.4189 | 0.4189 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | gan | Indeterminate (insufficient epochs) | None | 0.4686 | 0.4686 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | baseline | Underfitting | Lowered weight decay and boosted learning rate (1.5e-3) | 0.6386 | 0.6392 | +0.0006 |
| Model 3: BERT-style Transformer Encoder + NCF | smote | Indeterminate (insufficient epochs) | None | 0.4073 | 0.4073 | +0.0000 |
| Model 3: BERT-style Transformer Encoder + NCF | adasyn | Indeterminate (insufficient epochs) | None | 0.6251 | 0.6251 | +0.0000 |
| Model 3: BERT-style Transformer Encoder + NCF | gan | Indeterminate (insufficient epochs) | None | 0.6212 | 0.6212 | +0.0000 |
| Model 3: BERT-style Transformer Encoder + NCF | baseline | Underfitting | Lowered weight decay and boosted learning rate (1.5e-3) | 0.6384 | 0.6370 | -0.0014 |
| Model 4: Autoencoder + Recommender Network | smote | Indeterminate (insufficient epochs) | None | 0.5027 | 0.5027 | +0.0000 |
| Model 4: Autoencoder + Recommender Network | adasyn | Indeterminate (insufficient epochs) | None | 0.4996 | 0.4996 | +0.0000 |
| Model 4: Autoencoder + Recommender Network | gan | Indeterminate (insufficient epochs) | None | 0.4799 | 0.4799 | +0.0000 |
| Model 4: Autoencoder + Recommender Network | baseline | Healthy Fit / Well-Regularized | None | 0.5125 | 0.5125 | +0.0000 |
| Model 5: CNN + LSTM | smote | Indeterminate (insufficient epochs) | None | 0.9098 | 0.9098 | +0.0000 |
| Model 5: CNN + LSTM | adasyn | Indeterminate (insufficient epochs) | None | 0.8709 | 0.8709 | +0.0000 |
| Model 5: CNN + LSTM | gan | Indeterminate (insufficient epochs) | None | 0.8362 | 0.8362 | +0.0000 |
| Model 5: CNN + LSTM | baseline | Healthy Fit / Well-Regularized | None | 0.6451 | 0.6451 | +0.0000 |

## Table H: Validation vs. Final Frozen Test Performance Comparison (Knowledge Tracing)
| Model | Condition | Val Accuracy | Test Accuracy | Val ROC-AUC | Test ROC-AUC | Generalization Gap (AUC) |
|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | smote | 0.6832 | 0.5965 | 0.4736 | 0.4495 | +0.0241 |
| Model 1: LSTM + Attention | adasyn | 0.6832 | 0.5965 | 0.4569 | 0.4425 | +0.0144 |
| Model 1: LSTM + Attention | gan | 0.6832 | 0.5965 | 0.4451 | 0.4309 | +0.0142 |
| Model 2: Transformer + Knowledge Tracing | smote | 0.6832 | 0.5965 | 0.4228 | 0.4492 | -0.0264 |
| Model 2: Transformer + Knowledge Tracing | adasyn | 0.6832 | 0.5965 | 0.4189 | 0.4648 | -0.0459 |
| Model 2: Transformer + Knowledge Tracing | gan | 0.6832 | 0.5965 | 0.4686 | 0.4751 | -0.0064 |
| Model 3: BERT-style Transformer Encoder + NCF | smote | 0.6832 | 0.5965 | 0.4073 | 0.4198 | -0.0124 |
| Model 3: BERT-style Transformer Encoder + NCF | adasyn | 0.6832 | 0.6053 | 0.6251 | 0.5719 | +0.0532 |
| Model 3: BERT-style Transformer Encoder + NCF | gan | 0.6832 | 0.5965 | 0.6212 | 0.4703 | +0.1509 |
| Model 5: CNN + LSTM | smote | 0.8634 | 0.8246 | 0.9098 | 0.9453 | -0.0355 |
| Model 5: CNN + LSTM | adasyn | 0.8385 | 0.8860 | 0.8709 | 0.9268 | -0.0558 |
| Model 5: CNN + LSTM | gan | 0.7143 | 0.8158 | 0.8362 | 0.9220 | -0.0858 |
| Model 5: CNN + LSTM | baseline | 0.6221 | 0.6220 | 0.6451 | 0.6450 | +0.0001 |
| Model 1: LSTM + Attention | baseline | 0.6226 | 0.6229 | 0.6457 | 0.6456 | +0.0001 |
| Model 2: Transformer + Knowledge Tracing | baseline | 0.6180 | 0.6181 | 0.6392 | 0.6396 | -0.0004 |
| Model 3: BERT-style Transformer Encoder + NCF | baseline | 0.6165 | 0.6168 | 0.6370 | 0.6371 | -0.0000 |

## Table I: Core Discriminative Metrics on Unseen Test Cohort (Knowledge Tracing)
| Model | Strategy | Augmentation | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Log Loss |
|---|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | 10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4604 | 0.5438 | 0.6832 |
| Model 1: LSTM + Attention | 10fold | baseline | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4482 | 0.5339 | 0.7274 |
| Model 1: LSTM + Attention | 10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.5684 | 0.6675 | 0.6702 |
| Model 1: LSTM + Attention | 10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4434 | 0.5352 | 0.7876 |
| Model 1: LSTM + Attention | no_10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4425 | 0.5357 | 0.7568 |
| Model 1: LSTM + Attention | no_10fold | baseline | 0.6229 | 0.6427 | 0.7807 | 0.7050 | 0.6456 | 0.7009 | 0.6486 |
| Model 1: LSTM + Attention | no_10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4309 | 0.5294 | 0.7199 |
| Model 1: LSTM + Attention | no_10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4495 | 0.5379 | 0.6996 |
| Model 2: Transformer + Knowledge Tracing | 10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.5121 | 0.6240 | 0.6749 |
| Model 2: Transformer + Knowledge Tracing | 10fold | baseline | 0.5439 | 0.5909 | 0.7647 | 0.6667 | 0.4578 | 0.6017 | 0.7599 |
| Model 2: Transformer + Knowledge Tracing | 10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4425 | 0.5426 | 0.6882 |
| Model 2: Transformer + Knowledge Tracing | 10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4431 | 0.5669 | 0.6866 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4648 | 0.5775 | 0.7106 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | baseline | 0.6181 | 0.6362 | 0.7903 | 0.7049 | 0.6396 | 0.6962 | 0.6518 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4751 | 0.5859 | 0.6890 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4492 | 0.5692 | 0.6939 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | adasyn | 0.5614 | 0.5818 | 0.9412 | 0.7191 | 0.4492 | 0.5591 | 0.7184 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | baseline | 0.5965 | 0.6250 | 0.8088 | 0.7051 | 0.5377 | 0.6102 | 0.7592 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.3827 | 0.5825 | 0.7067 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | smote | 0.6053 | 0.6018 | 1.0000 | 0.7514 | 0.6049 | 0.7154 | 0.6640 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | adasyn | 0.6053 | 0.6018 | 1.0000 | 0.7514 | 0.5719 | 0.6678 | 0.6730 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | baseline | 0.6168 | 0.6416 | 0.7614 | 0.6964 | 0.6371 | 0.6934 | 0.6528 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4703 | 0.5967 | 0.6893 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4198 | 0.5794 | 0.7056 |
| Model 5: CNN + LSTM | 10fold | adasyn | 0.8246 | 0.9444 | 0.7500 | 0.8361 | 0.8101 | 0.9097 | 0.7016 |
| Model 5: CNN + LSTM | 10fold | baseline | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4463 | 0.5351 | 0.6958 |
| Model 5: CNN + LSTM | 10fold | gan | 0.8333 | 0.7882 | 0.9853 | 0.8758 | 0.8175 | 0.7743 | 1.0802 |
| Model 5: CNN + LSTM | 10fold | smote | 0.8684 | 0.8354 | 0.9706 | 0.8980 | 0.9584 | 0.9722 | 0.2483 |
| Model 5: CNN + LSTM | no_10fold | adasyn | 0.8860 | 0.8481 | 0.9853 | 0.9116 | 0.9268 | 0.9393 | 0.7646 |
| Model 5: CNN + LSTM | no_10fold | baseline | 0.6220 | 0.6414 | 0.7829 | 0.7051 | 0.6450 | 0.7009 | 0.6488 |
| Model 5: CNN + LSTM | no_10fold | gan | 0.8158 | 0.9434 | 0.7353 | 0.8264 | 0.9220 | 0.9544 | 0.3234 |
| Model 5: CNN + LSTM | no_10fold | smote | 0.8246 | 0.8750 | 0.8235 | 0.8485 | 0.9453 | 0.9658 | 0.2852 |

## Table J: Engineering Efficiency & Resource Utilization (Full-Test vs. Per-Batch Latency)
| Model | Train Time (s) | Full-Test Eval (s) | Per-Batch Latency (ms) | Throughput (evts/s) | Peak RAM (MB) | Peak GPU VRAM (MB) |
|---|---|---|---|---|---|---|
| **Model 5: CNN + LSTM** | 167.47 s | 4.0948 s | 2.29 ms | 226,621.2 | 8236.5 MB | 53.1 MB |
| **Model 1: LSTM + Attention** | 171.53 s | 4.1400 s | 2.31 ms | 224,144.3 | 8086.9 MB | 52.3 MB |
| **Model 2: Transformer + Knowledge Tracing** | 188.33 s | 4.5035 s | 2.51 ms | 206,054.5 | 8120.3 MB | 46.8 MB |
| **Model 4: Autoencoder + Recommender Network** | 138.63 s | 5.1717 s | 2.79 ms | 11,476.5 | 8093.3 MB | 106.1 MB |
| **Model 3: BERT-style Transformer Encoder + NCF** | 205.45 s | 4.6935 s | 2.62 ms | 197,710.9 | 8132.8 MB | 68.7 MB |

## Table K: Dedicated Recommendation Benchmark (Model 4: Autoencoder Recommender)
| Strategy | Augmentation | HitRate@5 | NDCG@5 | Precision@5 | Recall@5 | Reconstruction MSE | Full-Test Eval (s) | Per-Batch Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| 10fold | ADASYN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1347 | 0.0149 s | 14.90 ms |
| 10fold | BASELINE | 1.0000 | 0.5388 | 0.2667 | 0.3987 | 0.1406 | 0.0094 s | 9.40 ms |
| 10fold | GAN | 0.6667 | 0.2183 | 0.2000 | 0.0781 | 0.1607 | 0.0077 s | 7.70 ms |
| 10fold | SMOTE | 0.3333 | 0.1131 | 0.0667 | 0.0175 | 0.2333 | 0.0073 s | 7.30 ms |
| no_10fold | ADASYN | 0.3333 | 0.1617 | 0.1333 | 0.0351 | 0.1935 | 0.0136 s | 13.60 ms |
| no_10fold | BASELINE | 0.2260 | 0.0783 | 0.0695 | 0.0537 | 0.0178 | 5.1717 s | 2.79 ms |
| no_10fold | GAN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1586 | 0.0123 s | 12.30 ms |
| no_10fold | SMOTE | 0.3333 | 0.0713 | 0.0667 | 0.0303 | 0.2226 | 0.0098 s | 9.80 ms |