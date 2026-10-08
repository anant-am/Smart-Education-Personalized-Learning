# Comprehensive Codebase & Methodology Audit Report
**Project:** AI-Based Personalized Learning Recommendation System – Smart Education  
**Date:** September 12, 2026  
**Auditor:** Antigravity Autonomous Pair-Programming Agent  
**Environment:** Windows 11, Intel Core i5-13450HX, 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM), CUDA 12.6, PyTorch 2.14.0+cu126.

---

## 1. Executive Summary

This audit evaluates the codebase of the Smart Education project against the rigorous academic criteria specified by the faculty guidelines and master directives. The project investigates five hybrid deep-learning models for Knowledge Tracing and Educational Recommendation trained on the EdNet-KT3 dataset.

While previous development established initial working prototypes, this audit reveals critical methodological shortcomings that violate strict academic standards:
1. **Cohort Split & Directory Misalignment:** Previous runs performed a chronological 70/15/15 intra-user split across the 1,000 users in `KT-3 (1000)`, leaving the official `KT-3 (250) TEST` folder unutilized as the final held-out cohort.
2. **Trainer Object Reference Bug in Retraining:** In all 5 training scripts, `corrected_model` was instantiated and modified via `apply_correction()`, but `trainer.train(...)` was invoked on the original `trainer` instance (which still referenced `initial_model`). Consequently, the initial model was trained twice, and the corrected architecture was never retrained.
3. **Data Scaler & Vocabulary Leakage:** Encoders (question indices, concept tags) and continuous scalers (response time, inter-event time) were fit on the entire mixed dataset rather than strictly on the 1,000-user development cohort before transforming the 250-user test cohort.
4. **Directory Structure Verification:** Programmatic inspection revealed `A:\edge download\EdNet-KT3\KT3` contains 297,915 CSV files (~300k users), which would exhaust system RAM (16GB) if loaded indiscriminately. The development cohort `KT-3 (1000)` (1,000 users) and final unseen test cohort `KT-3 (250) TEST` (250 users) have zero student overlap and are perfectly suited for controlled, defensible academic experimentation.

All findings, discrepancies, and required corrective actions are detailed below.

---

## 2. Directory & Cohort Inspection

### 2.1 Actual Directory Findings
Programmatic verification via `pathlib` and Python confirmed the following:

| Path | Item Count | File Types | Role in Academic Pipeline | Audit Verdict |
|---|---|---|---|---|
| `A:\edge download\EdNet-KT3\KT-3 (1000)` | 1,000 files | CSV (`u1.csv` to `u1000.csv`) | **Development Cohort** (Training, Validation, 10-Fold CV, Hyperparameter Tuning) | **Compliant**: 1,000 valid user interaction logs. |
| `A:\edge download\EdNet-KT3\KT-3 (250) TEST` | 250 files | CSV (`u1165.csv` to `u1414.csv`) | **Final Unseen Test Cohort** (Completely untouched until final evaluation) | **Identified**: Isolated 250 test users. Zero overlap with `KT-3 (1000)`. |
| `A:\edge download\EdNet-KT3\KT3` | 297,915 files | CSV | Full raw EdNet-KT3 dataset (~300,000 learners) | **Out of Scope for Single-GPU Training**: Loading 298k users requires distributed clusters (>128GB RAM). Must remain strictly unreferenced. |
| `A:\edge download\EdNet-Contents\contents` | 4 CSV files | CSV (`questions.csv`, `lectures.csv`, `coupons.csv`, `payments.csv`) | Educational content metadata and relational hierarchy | **Compliant**: Contains 13,169 questions and 1,023 lectures. |

### 2.2 Student Overlap Analysis
Set intersection analysis between the student IDs in `KT-3 (1000)` and `KT-3 (250) TEST`:
$$\text{Overlap} = \{u \in \text{Dev}\} \cap \{u \in \text{Test}\} = \emptyset \quad (\text{Count} = 0)$$
The two cohorts are strictly disjoint, providing zero cross-user contamination.

---

## 3. Methodological Audit & Discrepancies

### 3.1 Preprocessing and Feature Extraction
- **Previous State:** `src/preprocessing.py` loaded only `KT-3 (1000)` and fit scalers (mean and standard deviation of `response_time_ms` and `time_since_prev_ms`) and index mappings (`question_to_idx`, `tag_to_idx`) across all rows before splitting.
- **Defect:** Scalers and encoders must be fitted **exclusively on the 1,000-user Development Cohort**. The 250-user Test Cohort must be transformed using development-fitted statistics, mapping unseen items/tags to the reserved padding/unknown index `0`.
- **Event Reconstruction:** `reconstruct_events_for_user` correctly processes raw interaction streams `enter(bundle) -> respond(question) -> submit(bundle)` and handles lecture engagement `enter(lecture) -> quit(lecture)`. This logic is mathematically sound and preserves the multi-action EdNet hierarchy.

### 3.2 Target and Temporal Leakage
- **Target Leakage:** `correct_answer` and `user_answer` were correctly flagged by `src/data_pipeline.py` and excluded from model inputs. The target $y \in \{0, 1\}$ (`is_correct`) is derived exclusively from $\mathbb{I}[\text{user\_answer} = \text{correct\_answer}]$.
- **Temporal Leakage:** Within the 1,000-user development cohort, sequences are ordered chronologically by `enter_ts`. Lag features (`previous_accuracy`, `recent_accuracy_5`, `attempt_count`) rely strictly on past history $t < \tau$, with zero future lookahead.
- **Evaluation Leakage:** The final test evaluation will now run strictly on the 250 unseen learners in `KT-3 (250) TEST`, which have never been observed during training, validation, or hyperparameter selection.

### 3.3 10-Fold Cross-Validation & SMOTE Compliance
- **Cross-Validation:** 10-fold grouped cross-validation (`GroupKFold` grouped by `user_id`) must be executed on the 1,000-user Development Cohort to establish statistically robust performance without cross-fold student leakage.
- **SMOTE Boundary:** SMOTE is applied strictly to flat feature vectors of the training folds within the Development Cohort to balance class representations for tabular learning gap analysis. SMOTE is **never** applied to temporal sequence inputs, validation folds, or the 250-user unseen test cohort.

### 3.4 Model Retraining Bug in `train_model*.py`
- **Root Cause Analysis:** In the previous scripts:
  ```python
  trainer = ModelTrainer(initial_model, device)
  initial_history = trainer.train(...)
  diagnosis, evidence = trainer.detect_overfitting(initial_history)
  corrected_model = copy.deepcopy(initial_model)
  corrected_model, correction_info = trainer.apply_correction(corrected_model, diagnosis)
  # BUG: trainer still holds reference to initial_model!
  retrained_history = trainer.train(...) 
  ```
- **Consequence:** `trainer.model` continued pointing to `initial_model`. The modified `corrected_model` remained un-trained, and the initial model was simply trained for 8 additional epochs.
- **Remediation:** A fresh `ModelTrainer` instance must be instantiated explicitly for the corrected model:
  ```python
  corrected_trainer = ModelTrainer(corrected_model, device)
  retrained_history = corrected_trainer.train(...)
  ```

---

## 4. Five Hybrid Model Architectures Audit

The project requires five hybrid deep-learning models:

| # | Model Architecture | Components | File Location | Audit Status |
|---|---|---|---|---|
| 1 | **Transformer + Knowledge Tracing** | Sinusoidal Positional Encoding + Multi-Head Self-Attention + Causal Upper-Triangular Mask + Feed-Forward Network | `src/models.py` (`TransformerKT`) | Verified. Need `-1e4` causal mask constant to avoid AMP `c10::Half` overflow. |
| 2 | **LSTM + Attention** | Item/Concept Embeddings + 1-Layer LSTM + Scaled Dot-Product Attention + Linear Logit Head | `src/models.py` (`LSTMAttentionKT`) | Verified. Attention weights extractable for interpretability. |
| 3 | **BERT + Neural Collaborative Filtering** | Bidirectional/Causal Attention Encoder + Generalized Matrix Factorization (GMF) + Multi-Layer Perceptron (MLP) Fusion | `src/models.py` (`BERTNCF`) | Verified. Supports user-state to target-item matching. |
| 4 | **Autoencoder + Recommender Network** | Bottleneck Encoder (latent dim 32) + Reconstruction Decoder + Dense Recommendation Head | `src/models.py` (`AutoencoderRecommender`) | Verified. Evaluates reconstruction MSE and recommendation BCE simultaneously. |
| 5 | **CNN + LSTM** | 1D Temporal Convolution (kernel size 3) + BatchNorm1d + LSTM Sequence Tracker + Dense Logit Head | `src/models.py` (`CNNLSTM`) | Verified. Captures local temporal n-gram patterns and global state dynamics. |

---

## 5. Action Plan & Required Code Modifications

1. **`config.py`**:
   - Define `EDNET_KT3_DEV_DIR = Path(r"A:\edge download\EdNet-KT3\KT-3 (1000)")`.
   - Define `EDNET_KT3_TEST_DIR = Path(r"A:\edge download\EdNet-KT3\KT-3 (250) TEST")`.
   - Document `EDNET_KT3_FULL_DIR = Path(r"A:\edge download\EdNet-KT3\KT3")` as inspected and quarantined.
2. **`src/preprocessing.py`**:
   - Process `KT-3 (1000)` to generate `train_dev.csv`. Fit `question_to_idx`, `tag_to_idx`, `source_map`, `platform_map`, and continuous feature scalers exclusively on `KT-3 (1000)`.
   - Transform `KT-3 (250) TEST` using strictly the development encoders/scalers to generate `test_unseen_250.csv`.
3. **`src/data_pipeline.py`**:
   - Document leakage checks in `reports/leakage_report.md`.
   - Run 10-fold `GroupKFold` on the 1,000 development users, saving `reports/cv_report.md`.
   - Apply SMOTE only to training folds, saving `reports/smote_report.md`.
4. **`src/trainer.py` & `scripts/train_model*.py`**:
   - Fix the `ModelTrainer` instance bug across all 5 training scripts.
   - Train initial model -> detect overfitting -> apply regularizing/capacity correction -> retrain fresh `corrected_trainer` -> evaluate on `test_unseen_250.csv`.
5. **Literature Mapping**:
   - Synthesize research directions from the foundational EdNet and Knowledge Tracing literature into `reports/future_work_mapping.md`.
6. **Model Comparison & Final Reports**:
   - Generate `reports/model_comparison.md`, `reports/final_results.md`, and comparative loss/ROC plots.
