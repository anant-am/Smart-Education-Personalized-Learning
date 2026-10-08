# Comprehensive Experiment Matrix Benchmark Summary
**Project:** AI-Based Personalized Learning Recommendation System — Smart Education  
**Execution Timestamp:** 2026-10-08 23:07:58  
**Execution Device:** cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU)  
**Evaluated Configurations:** 40 primary configurations  

## Methodological Overview & Safeguards
- **Strict User-Level Split:** All students partitioned deterministically into 80% Train/Dev and 20% Final Test cohorts. $\text{Train} \cap \text{Test} = \emptyset$.
- **Frozen Test Cohort:** The 20% test partition (`test.csv`) was strictly frozen; no SMOTE, ADASYN, or GAN augmentation ever touched the test or validation cohorts.
- **Retraining Invariant:** Fresh model and optimizer instantiated for all correction retrainings.
- **No Fabrication:** All reported scores, latencies, and resource usages are measured empirically.

## Table A: Architectural Model Comparison (Baseline, No 10-Fold)
| Model | Parameters | Val Accuracy | Val ROC-AUC | Val F1 | Test Accuracy | Test ROC-AUC | Test F1 | Train Time (s) | Inference (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Model 1: LSTM + Attention** | 815,361 | 0.6832 | 0.3738 | 0.8118 | **0.5965** | **0.4345** | **0.7473** | 2.68 | 0.0209 |
| **Model 2: Transformer + Knowledge Tracing** | 836,545 | 0.6708 | 0.3820 | 0.8030 | **0.5877** | **0.4322** | **0.7345** | 0.11 | 0.0105 |
| **Model 3: BERT-style Transformer Encoder + NCF** | 1,597,249 | 0.5590 | 0.4922 | 0.6900 | **0.5614** | **0.5518** | **0.6528** | 0.07 | 0.0176 |
| **Model 5: CNN + LSTM** | 815,361 | 0.6273 | 0.5002 | 0.7600 | **0.6228** | **0.5384** | **0.7425** | 0.33 | 0.0201 |
| **Model 4: Autoencoder + Recommender Network** | 25,768 | 0.4750 | 0.5246 | 0.4101 | **0.4500** | **0.5132** | **0.1042** | 1.59 | 0.0102 |

## Table B: Validation Strategy Comparison (No 10-Fold vs. 10-Fold CV on Baseline)
| Model | Strategy | Val Accuracy | Val ROC-AUC | Val F1 | Test Accuracy | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | 10fold | 0.6832 | 0.4795 | 0.8118 | **0.5965** | **0.4482** |
| Model 1: LSTM + Attention | no_10fold | 0.6832 | 0.3738 | 0.8118 | **0.5965** | **0.4345** |
| Model 2: Transformer + Knowledge Tracing | 10fold | 0.6646 | 0.4963 | 0.7939 | **0.5439** | **0.4578** |
| Model 2: Transformer + Knowledge Tracing | no_10fold | 0.6708 | 0.3820 | 0.8030 | **0.5877** | **0.4322** |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | 0.6770 | 0.5923 | 0.7815 | **0.5965** | **0.5377** |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | 0.5590 | 0.4922 | 0.6900 | **0.5614** | **0.5518** |
| Model 4: Autoencoder + Recommender Network | 10fold | 0.4875 | 0.4921 | 0.3807 | **0.5320** | **0.4954** |
| Model 4: Autoencoder + Recommender Network | no_10fold | 0.4750 | 0.5246 | 0.4101 | **0.4500** | **0.5132** |
| Model 5: CNN + LSTM | 10fold | 0.6708 | 0.4742 | 0.8015 | **0.5965** | **0.4463** |
| Model 5: CNN + LSTM | no_10fold | 0.6273 | 0.5002 | 0.7600 | **0.6228** | **0.5384** |

## Table C: SMOTE Ablation Benchmark (Baseline vs. SMOTE)
| Model | Condition | Val Acc | Val AUC | Val Recall | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | **BASELINE** | 0.6832 | 0.3738 | 1.0000 | 0.5965 | 0.4345 | **1.0000** | **0.7473** |
| Model 1: LSTM + Attention | **SMOTE** | 0.6832 | 0.4736 | 1.0000 | 0.5965 | 0.4495 | **1.0000** | **0.7473** |
| Model 2: Transformer + Knowledge Tracing | **BASELINE** | 0.6708 | 0.3820 | 0.9818 | 0.5877 | 0.4322 | **0.9559** | **0.7345** |
| Model 2: Transformer + Knowledge Tracing | **SMOTE** | 0.6832 | 0.4228 | 1.0000 | 0.5965 | 0.4492 | **1.0000** | **0.7473** |
| Model 3: BERT-style Transformer Encoder + NCF | **BASELINE** | 0.5590 | 0.4922 | 0.7182 | 0.5614 | 0.5518 | **0.6912** | **0.6528** |
| Model 3: BERT-style Transformer Encoder + NCF | **SMOTE** | 0.6832 | 0.4073 | 1.0000 | 0.5965 | 0.4198 | **1.0000** | **0.7473** |
| Model 4: Autoencoder + Recommender Network | **BASELINE** | 0.4750 | 0.5246 | 0.5935 | 0.4500 | 0.5132 | **0.5161** | **0.1042** |
| Model 4: Autoencoder + Recommender Network | **SMOTE** | 0.5075 | 0.5027 | 0.5122 | 0.4340 | 0.4631 | **0.4839** | **0.0958** |
| Model 5: CNN + LSTM | **BASELINE** | 0.6273 | 0.5002 | 0.8636 | 0.6228 | 0.5384 | **0.9118** | **0.7425** |
| Model 5: CNN + LSTM | **SMOTE** | 0.8634 | 0.9098 | 0.9727 | 0.8246 | 0.9453 | **0.8235** | **0.8485** |

## Table D: ADASYN Ablation Benchmark (Baseline vs. ADASYN)
| Model | Condition | Val Acc | Val AUC | Val Recall | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | **ADASYN** | 0.6832 | 0.4569 | 1.0000 | 0.5965 | 0.4425 | **1.0000** | **0.7473** |
| Model 1: LSTM + Attention | **BASELINE** | 0.6832 | 0.3738 | 1.0000 | 0.5965 | 0.4345 | **1.0000** | **0.7473** |
| Model 2: Transformer + Knowledge Tracing | **ADASYN** | 0.6832 | 0.4189 | 1.0000 | 0.5965 | 0.4648 | **1.0000** | **0.7473** |
| Model 2: Transformer + Knowledge Tracing | **BASELINE** | 0.6708 | 0.3820 | 0.9818 | 0.5877 | 0.4322 | **0.9559** | **0.7345** |
| Model 3: BERT-style Transformer Encoder + NCF | **ADASYN** | 0.6832 | 0.6251 | 1.0000 | 0.6053 | 0.5719 | **1.0000** | **0.7514** |
| Model 3: BERT-style Transformer Encoder + NCF | **BASELINE** | 0.5590 | 0.4922 | 0.7182 | 0.5614 | 0.5518 | **0.6912** | **0.6528** |
| Model 4: Autoencoder + Recommender Network | **ADASYN** | 0.5125 | 0.4996 | 0.5366 | 0.4840 | 0.4902 | **0.4516** | **0.0979** |
| Model 4: Autoencoder + Recommender Network | **BASELINE** | 0.4750 | 0.5246 | 0.5935 | 0.4500 | 0.5132 | **0.5161** | **0.1042** |
| Model 5: CNN + LSTM | **ADASYN** | 0.8385 | 0.8709 | 0.9818 | 0.8860 | 0.9268 | **0.9853** | **0.9116** |
| Model 5: CNN + LSTM | **BASELINE** | 0.6273 | 0.5002 | 0.8636 | 0.6228 | 0.5384 | **0.9118** | **0.7425** |

## Table E: Generative GAN Ablation Benchmark (Baseline vs. Tabular WGAN-GP)
| Model | Condition | Val Acc | Val AUC | Val Recall | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | **BASELINE** | 0.6832 | 0.3738 | 1.0000 | 0.5965 | 0.4345 | **1.0000** | **0.7473** |
| Model 1: LSTM + Attention | **GAN** | 0.6832 | 0.4451 | 1.0000 | 0.5965 | 0.4309 | **1.0000** | **0.7473** |
| Model 2: Transformer + Knowledge Tracing | **BASELINE** | 0.6708 | 0.3820 | 0.9818 | 0.5877 | 0.4322 | **0.9559** | **0.7345** |
| Model 2: Transformer + Knowledge Tracing | **GAN** | 0.6832 | 0.4686 | 1.0000 | 0.5965 | 0.4751 | **1.0000** | **0.7473** |
| Model 3: BERT-style Transformer Encoder + NCF | **BASELINE** | 0.5590 | 0.4922 | 0.7182 | 0.5614 | 0.5518 | **0.6912** | **0.6528** |
| Model 3: BERT-style Transformer Encoder + NCF | **GAN** | 0.6832 | 0.6212 | 1.0000 | 0.5965 | 0.4703 | **1.0000** | **0.7473** |
| Model 4: Autoencoder + Recommender Network | **BASELINE** | 0.4750 | 0.5246 | 0.5935 | 0.4500 | 0.5132 | **0.5161** | **0.1042** |
| Model 4: Autoencoder + Recommender Network | **GAN** | 0.5050 | 0.4799 | 0.4634 | 0.5140 | 0.4206 | **0.4194** | **0.0967** |
| Model 5: CNN + LSTM | **BASELINE** | 0.6273 | 0.5002 | 0.8636 | 0.6228 | 0.5384 | **0.9118** | **0.7425** |
| Model 5: CNN + LSTM | **GAN** | 0.7143 | 0.8362 | 0.6364 | 0.8158 | 0.9220 | **0.7353** | **0.8264** |

## Table F: Comprehensive Resampling Paradigm Benchmark (All 4 Augmentation Conditions)
| Model | Augmentation | Val Acc | Val AUC | Test Acc | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | ADASYN | 0.6832 | 0.4569 | 0.5965 | 0.4425 | 1.0000 | 0.7473 |
| Model 1: LSTM + Attention | BASELINE | 0.6832 | 0.3738 | 0.5965 | 0.4345 | 1.0000 | 0.7473 |
| Model 1: LSTM + Attention | GAN | 0.6832 | 0.4451 | 0.5965 | 0.4309 | 1.0000 | 0.7473 |
| Model 1: LSTM + Attention | SMOTE | 0.6832 | 0.4736 | 0.5965 | 0.4495 | 1.0000 | 0.7473 |
| Model 2: Transformer + Knowledge Tracing | ADASYN | 0.6832 | 0.4189 | 0.5965 | 0.4648 | 1.0000 | 0.7473 |
| Model 2: Transformer + Knowledge Tracing | BASELINE | 0.6708 | 0.3820 | 0.5877 | 0.4322 | 0.9559 | 0.7345 |
| Model 2: Transformer + Knowledge Tracing | GAN | 0.6832 | 0.4686 | 0.5965 | 0.4751 | 1.0000 | 0.7473 |
| Model 2: Transformer + Knowledge Tracing | SMOTE | 0.6832 | 0.4228 | 0.5965 | 0.4492 | 1.0000 | 0.7473 |
| Model 3: BERT-style Transformer Encoder + NCF | ADASYN | 0.6832 | 0.6251 | 0.6053 | 0.5719 | 1.0000 | 0.7514 |
| Model 3: BERT-style Transformer Encoder + NCF | BASELINE | 0.5590 | 0.4922 | 0.5614 | 0.5518 | 0.6912 | 0.6528 |
| Model 3: BERT-style Transformer Encoder + NCF | GAN | 0.6832 | 0.6212 | 0.5965 | 0.4703 | 1.0000 | 0.7473 |
| Model 3: BERT-style Transformer Encoder + NCF | SMOTE | 0.6832 | 0.4073 | 0.5965 | 0.4198 | 1.0000 | 0.7473 |
| Model 4: Autoencoder + Recommender Network | ADASYN | 0.5125 | 0.4996 | 0.4840 | 0.4902 | 0.4516 | 0.0979 |
| Model 4: Autoencoder + Recommender Network | BASELINE | 0.4750 | 0.5246 | 0.4500 | 0.5132 | 0.5161 | 0.1042 |
| Model 4: Autoencoder + Recommender Network | GAN | 0.5050 | 0.4799 | 0.5140 | 0.4206 | 0.4194 | 0.0967 |
| Model 4: Autoencoder + Recommender Network | SMOTE | 0.5075 | 0.5027 | 0.4340 | 0.4631 | 0.4839 | 0.0958 |
| Model 5: CNN + LSTM | ADASYN | 0.8385 | 0.8709 | 0.8860 | 0.9268 | 0.9853 | 0.9116 |
| Model 5: CNN + LSTM | BASELINE | 0.6273 | 0.5002 | 0.6228 | 0.5384 | 0.9118 | 0.7425 |
| Model 5: CNN + LSTM | GAN | 0.7143 | 0.8362 | 0.8158 | 0.9220 | 0.7353 | 0.8264 |
| Model 5: CNN + LSTM | SMOTE | 0.8634 | 0.9098 | 0.8246 | 0.9453 | 0.8235 | 0.8485 |

## Table G: Fit Diagnosis & Regularization Retraining Invariant
| Model | Augmentation | Initial Diagnosis | Correction Plan | Val AUC Before | Val AUC After | Delta AUC |
|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | baseline | Indeterminate (insufficient epochs) | None | 0.3738 | 0.3738 | +0.0000 |
| Model 1: LSTM + Attention | smote | Indeterminate (insufficient epochs) | None | 0.4736 | 0.4736 | +0.0000 |
| Model 1: LSTM + Attention | adasyn | Indeterminate (insufficient epochs) | None | 0.4569 | 0.4569 | +0.0000 |
| Model 1: LSTM + Attention | gan | Indeterminate (insufficient epochs) | None | 0.4451 | 0.4451 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | baseline | Indeterminate (insufficient epochs) | None | 0.3820 | 0.3820 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | smote | Indeterminate (insufficient epochs) | None | 0.4228 | 0.4228 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | adasyn | Indeterminate (insufficient epochs) | None | 0.4189 | 0.4189 | +0.0000 |
| Model 2: Transformer + Knowledge Tracing | gan | Indeterminate (insufficient epochs) | None | 0.4686 | 0.4686 | +0.0000 |
| Model 3: BERT-style Transformer Encoder + NCF | baseline | Indeterminate (insufficient epochs) | None | 0.4922 | 0.4922 | +0.0000 |
| Model 3: BERT-style Transformer Encoder + NCF | smote | Indeterminate (insufficient epochs) | None | 0.4073 | 0.4073 | +0.0000 |
| Model 3: BERT-style Transformer Encoder + NCF | adasyn | Indeterminate (insufficient epochs) | None | 0.6251 | 0.6251 | +0.0000 |
| Model 3: BERT-style Transformer Encoder + NCF | gan | Indeterminate (insufficient epochs) | None | 0.6212 | 0.6212 | +0.0000 |
| Model 4: Autoencoder + Recommender Network | baseline | Indeterminate (insufficient epochs) | None | 0.5246 | 0.5246 | +0.0000 |
| Model 4: Autoencoder + Recommender Network | smote | Indeterminate (insufficient epochs) | None | 0.5027 | 0.5027 | +0.0000 |
| Model 4: Autoencoder + Recommender Network | adasyn | Indeterminate (insufficient epochs) | None | 0.4996 | 0.4996 | +0.0000 |
| Model 4: Autoencoder + Recommender Network | gan | Indeterminate (insufficient epochs) | None | 0.4799 | 0.4799 | +0.0000 |
| Model 5: CNN + LSTM | baseline | Indeterminate (insufficient epochs) | None | 0.5002 | 0.5002 | +0.0000 |
| Model 5: CNN + LSTM | smote | Indeterminate (insufficient epochs) | None | 0.9098 | 0.9098 | +0.0000 |
| Model 5: CNN + LSTM | adasyn | Indeterminate (insufficient epochs) | None | 0.8709 | 0.8709 | +0.0000 |
| Model 5: CNN + LSTM | gan | Indeterminate (insufficient epochs) | None | 0.8362 | 0.8362 | +0.0000 |

## Table H: Validation vs. Final Frozen Test Performance Comparison
| Model | Condition | Val Accuracy | Test Accuracy | Val ROC-AUC | Test ROC-AUC | Generalization Gap (AUC) |
|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | baseline | 0.6832 | 0.5965 | 0.3738 | 0.4345 | -0.0607 |
| Model 1: LSTM + Attention | smote | 0.6832 | 0.5965 | 0.4736 | 0.4495 | +0.0241 |
| Model 1: LSTM + Attention | adasyn | 0.6832 | 0.5965 | 0.4569 | 0.4425 | +0.0144 |
| Model 1: LSTM + Attention | gan | 0.6832 | 0.5965 | 0.4451 | 0.4309 | +0.0142 |
| Model 2: Transformer + Knowledge Tracing | baseline | 0.6708 | 0.5877 | 0.3820 | 0.4322 | -0.0502 |
| Model 2: Transformer + Knowledge Tracing | smote | 0.6832 | 0.5965 | 0.4228 | 0.4492 | -0.0264 |
| Model 2: Transformer + Knowledge Tracing | adasyn | 0.6832 | 0.5965 | 0.4189 | 0.4648 | -0.0459 |
| Model 2: Transformer + Knowledge Tracing | gan | 0.6832 | 0.5965 | 0.4686 | 0.4751 | -0.0064 |
| Model 3: BERT-style Transformer Encoder + NCF | baseline | 0.5590 | 0.5614 | 0.4922 | 0.5518 | -0.0596 |
| Model 3: BERT-style Transformer Encoder + NCF | smote | 0.6832 | 0.5965 | 0.4073 | 0.4198 | -0.0124 |
| Model 3: BERT-style Transformer Encoder + NCF | adasyn | 0.6832 | 0.6053 | 0.6251 | 0.5719 | +0.0532 |
| Model 3: BERT-style Transformer Encoder + NCF | gan | 0.6832 | 0.5965 | 0.6212 | 0.4703 | +0.1509 |
| Model 5: CNN + LSTM | baseline | 0.6273 | 0.6228 | 0.5002 | 0.5384 | -0.0382 |
| Model 5: CNN + LSTM | smote | 0.8634 | 0.8246 | 0.9098 | 0.9453 | -0.0355 |
| Model 5: CNN + LSTM | adasyn | 0.8385 | 0.8860 | 0.8709 | 0.9268 | -0.0558 |
| Model 5: CNN + LSTM | gan | 0.7143 | 0.8158 | 0.8362 | 0.9220 | -0.0858 |
| Model 4: Autoencoder + Recommender Network | baseline | 0.4750 | 0.4500 | 0.5246 | 0.5132 | +0.0114 |
| Model 4: Autoencoder + Recommender Network | smote | 0.5075 | 0.4340 | 0.5027 | 0.4631 | +0.0396 |
| Model 4: Autoencoder + Recommender Network | adasyn | 0.5125 | 0.4840 | 0.4996 | 0.4902 | +0.0094 |
| Model 4: Autoencoder + Recommender Network | gan | 0.5050 | 0.5140 | 0.4799 | 0.4206 | +0.0593 |

## Table I: Core Discriminative Metrics on Unseen Test Cohort
| Model | Strategy | Augmentation | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Log Loss |
|---|---|---|---|---|---|---|---|---|---|
| Model 1: LSTM + Attention | 10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4604 | 0.5438 | 0.6832 |
| Model 1: LSTM + Attention | 10fold | baseline | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4482 | 0.5339 | 0.7274 |
| Model 1: LSTM + Attention | 10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.5684 | 0.6675 | 0.6702 |
| Model 1: LSTM + Attention | 10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4434 | 0.5352 | 0.7876 |
| Model 1: LSTM + Attention | no_10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4425 | 0.5357 | 0.7568 |
| Model 1: LSTM + Attention | no_10fold | baseline | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4345 | 0.5204 | 0.6849 |
| Model 1: LSTM + Attention | no_10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4309 | 0.5294 | 0.7199 |
| Model 1: LSTM + Attention | no_10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4495 | 0.5379 | 0.6996 |
| Model 2: Transformer + Knowledge Tracing | 10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.5121 | 0.6240 | 0.6749 |
| Model 2: Transformer + Knowledge Tracing | 10fold | baseline | 0.5439 | 0.5909 | 0.7647 | 0.6667 | 0.4578 | 0.6017 | 0.7599 |
| Model 2: Transformer + Knowledge Tracing | 10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4425 | 0.5426 | 0.6882 |
| Model 2: Transformer + Knowledge Tracing | 10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4431 | 0.5669 | 0.6866 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | adasyn | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4648 | 0.5775 | 0.7106 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | baseline | 0.5877 | 0.5963 | 0.9559 | 0.7345 | 0.4322 | 0.5694 | 0.9492 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4751 | 0.5859 | 0.6890 |
| Model 2: Transformer + Knowledge Tracing | no_10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4492 | 0.5692 | 0.6939 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | adasyn | 0.5614 | 0.5818 | 0.9412 | 0.7191 | 0.4492 | 0.5591 | 0.7184 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | baseline | 0.5965 | 0.6250 | 0.8088 | 0.7051 | 0.5377 | 0.6102 | 0.7592 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.3827 | 0.5825 | 0.7067 |
| Model 3: BERT-style Transformer Encoder + NCF | 10fold | smote | 0.6053 | 0.6018 | 1.0000 | 0.7514 | 0.6049 | 0.7154 | 0.6640 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | adasyn | 0.6053 | 0.6018 | 1.0000 | 0.7514 | 0.5719 | 0.6678 | 0.6730 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | baseline | 0.5614 | 0.6184 | 0.6912 | 0.6528 | 0.5518 | 0.6354 | 0.7572 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | gan | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4703 | 0.5967 | 0.6893 |
| Model 3: BERT-style Transformer Encoder + NCF | no_10fold | smote | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4198 | 0.5794 | 0.7056 |
| Model 4: Autoencoder + Recommender Network | 10fold | adasyn | 0.5460 | 0.0664 | 0.4839 | 0.1167 | 0.4802 | 0.0827 | 0.1347 |
| Model 4: Autoencoder + Recommender Network | 10fold | baseline | 0.5320 | 0.0644 | 0.4839 | 0.1136 | 0.4954 | 0.1564 | 0.1406 |
| Model 4: Autoencoder + Recommender Network | 10fold | gan | 0.4800 | 0.0712 | 0.6129 | 0.1275 | 0.6126 | 0.1612 | 0.1607 |
| Model 4: Autoencoder + Recommender Network | 10fold | smote | 0.4640 | 0.0691 | 0.6129 | 0.1242 | 0.5785 | 0.1441 | 0.2333 |
| Model 4: Autoencoder + Recommender Network | no_10fold | adasyn | 0.4840 | 0.0549 | 0.4516 | 0.0979 | 0.4902 | 0.1168 | 0.1935 |
| Model 4: Autoencoder + Recommender Network | no_10fold | baseline | 0.4500 | 0.0580 | 0.5161 | 0.1042 | 0.5132 | 0.1201 | 0.1212 |
| Model 4: Autoencoder + Recommender Network | no_10fold | gan | 0.5140 | 0.0546 | 0.4194 | 0.0967 | 0.4206 | 0.0675 | 0.1586 |
| Model 4: Autoencoder + Recommender Network | no_10fold | smote | 0.4340 | 0.0532 | 0.4839 | 0.0958 | 0.4631 | 0.0941 | 0.2226 |
| Model 5: CNN + LSTM | 10fold | adasyn | 0.8246 | 0.9444 | 0.7500 | 0.8361 | 0.8101 | 0.9097 | 0.7016 |
| Model 5: CNN + LSTM | 10fold | baseline | 0.5965 | 0.5965 | 1.0000 | 0.7473 | 0.4463 | 0.5351 | 0.6958 |
| Model 5: CNN + LSTM | 10fold | gan | 0.8333 | 0.7882 | 0.9853 | 0.8758 | 0.8175 | 0.7743 | 1.0802 |
| Model 5: CNN + LSTM | 10fold | smote | 0.8684 | 0.8354 | 0.9706 | 0.8980 | 0.9584 | 0.9722 | 0.2483 |
| Model 5: CNN + LSTM | no_10fold | adasyn | 0.8860 | 0.8481 | 0.9853 | 0.9116 | 0.9268 | 0.9393 | 0.7646 |
| Model 5: CNN + LSTM | no_10fold | baseline | 0.6228 | 0.6263 | 0.9118 | 0.7425 | 0.5384 | 0.6308 | 0.6869 |
| Model 5: CNN + LSTM | no_10fold | gan | 0.8158 | 0.9434 | 0.7353 | 0.8264 | 0.9220 | 0.9544 | 0.3234 |
| Model 5: CNN + LSTM | no_10fold | smote | 0.8246 | 0.8750 | 0.8235 | 0.8485 | 0.9453 | 0.9658 | 0.2852 |

## Table J: Engineering Efficiency & Resource Utilization
| Model | Train Time (s) | Inference (s) | Throughput (evts/s) | Peak RAM (MB) | Peak GPU VRAM (MB) |
|---|---|---|---|---|---|
| **Model 1: LSTM + Attention** | 2.68 s | 0.0209 s | 36,788.8 | 2048.8 MB | 42.4 MB |
| **Model 2: Transformer + Knowledge Tracing** | 0.11 s | 0.0105 s | 73,122.2 | 2501.2 MB | 36.1 MB |
| **Model 3: BERT-style Transformer Encoder + NCF** | 0.07 s | 0.0176 s | 43,570.3 | 2520.3 MB | 54.3 MB |
| **Model 5: CNN + LSTM** | 0.33 s | 0.0201 s | 38,285.3 | 2539.6 MB | 42.4 MB |
| **Model 4: Autoencoder + Recommender Network** | 1.59 s | 0.0102 s | 490.1 | 1670.0 MB | 16.8 MB |