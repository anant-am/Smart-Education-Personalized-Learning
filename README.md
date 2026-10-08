# AI-Based Personalized Learning Recommendation System — Smart Education

**Academic AI/ML Project — University Defense Grade**  
**Execution Environment:** Windows 11, Intel Core i5-13450HX, 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM, CUDA 12.6, PyTorch 2.14.0+cu126)  
**Dataset:** EdNet-KT3 (1,000-User Development Cohort + 250-User Isolated Unseen Test Cohort) + EdNet-Contents  

---

## 1. System Architecture & Lifecycle

This project implements a multi-model personalized educational recommendation engine adhering to the sequential machine-learning lifecycle prescribed by university faculty guidelines:

```text
               ┌─────────────────────────────────────────────────┐
               │    EdNet-KT3 Raw Interaction Event Streams      │
               │   (enter -> respond -> submit -> quit actions)   │
               └───────────────────────┬─────────────────────────┘
                                       │
                                       ▼
               ┌─────────────────────────────────────────────────┐
               │   Event Reconstruction & Causal Feature Lagging │
               │   (Atomic Question Attempts, Durations, Prior)  │
               └───────────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌─────────────────────────┐           ┌─────────────────────────┐
        │  Development Cohort     │           │  Final Unseen Test Set  │
        │  KT-3 (1000) (1000 U)   │           │  KT-3 (250) TEST (250 U)│
        │  [Train 80% / Val 20%]  │           │  [STRICTLY UNTOUCHED]   │
        └───────────┬─────────────┘           └───────────┬─────────────┘
                    │ (10-Fold CV & SMOTE)                │
                    ▼                                     │
        ┌─────────────────────────────────────────┐       │
        │  5 Standardized Hybrid DL Architectures │       │
        │  Model 1: LSTM + Attention (KT)         │       │
        │  Model 2: Transformer + KT              │       │
        │  Model 3: BERT-style Transformer + NCF  │       │
        │  Model 4: Autoencoder + Recommender     │       │
        │  Model 5: CNN + LSTM                    │       │
        └───────────────────┬─────────────────────┘       │
                            │ (Initial -> Retrained)      │
                            ▼                             │
        ┌─────────────────────────────────────────────────┴─────┐
        │  Final Academic Evaluation on 249 Unseen Learners     │
        │  (Accuracy, AUC, F1, LogLoss, Precision@K, NDCG@K)    │
        └───────────────────┬───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────────────┐
        │  Personalized Recommendation Engine (Actual Catalog)  │
        │  (Direct remedial lectures & explanation targeting)   │
        └───────────────────────────────────────────────────────┘
```

---

## 2. Standardized Five Hybrid Deep-Learning Models

1. **Model 1 — LSTM + Attention (Knowledge Tracing):**
   Multi-feature item, concept tag, part, and continuous engagement projection through a recurrent LSTM layer, followed by scaled dot-product self-attention with causal lookahead masking.
2. **Model 2 — Transformer + Knowledge Tracing:**
   Multi-head causal self-attention network with sinusoidal positional encodings and temporal interval projections tracking multi-step mastery state transitions.
3. **Model 3 — BERT-style Transformer + Neural Collaborative Filtering (NCF):**
   Transformer sequence encoder deriving contextual student representations fused with target item embeddings via Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) branches into a NeuMF collaborative prediction.
4. **Model 4 — Autoencoder + Recommender Network:**
   Non-linear compression bottleneck autoencoder reconstructing sparse interaction vectors and scoring candidate learning resources (video lectures, explanations, practice questions) through a dedicated recommender head.
5. **Model 5 — CNN + LSTM:**
   1D temporal convolution extracting localized multi-step practice patterns combined with an LSTM layer tracking long-term cumulative learning trajectories.

---

## 3. Directory Layout & Cohort Partitioning

```text
A:\edge download\
│
├── EdNet-Contents\
│   └── contents\
│       ├── questions.csv        (13,169 items: bundle_id, explanation_id, correct_answer, part, tags)
│       └── lectures.csv         (1,021 items: part, tags, video_length, deployed_at)
│
├── EdNet-KT3\
│   ├── KT-3 (1000)\             (Development Cohort: exactly 1,000 students, 532,594 events)
│   ├── KT-3 (250) TEST\         (Final Unseen Test Cohort: exactly 250 students, 174,608 events)
│   └── KT3\                     (Full raw dataset: ~300k files; inspected & quarantined)
│
└── Smart_Education_Project\
    ├── config.py                (Central paths, hyperparameters, random seed 42)
    ├── data\processed\          (train_dev.csv, test_unseen_250.csv, train.csv, val.csv, test.csv)
    ├── models\checkpoints\      (Saved model checkpoints with paired JSON metadata)
    ├── plots\                   (Training curves and 5-model comparative visualizations)
    ├── reports\                 (14 mandatory academic defense reports and manifests)
    ├── scripts\                 (Standardized runners: prepare_data.py, run_model[1-5].py, run_all.py)
    └── src\                     (Modular implementations: models, dataset, trainer, cross_validation, smote_utils, leakage_checks, knowledge_state, recommendation)
```

---

## 4. Academic Methodology & Safeguards

- **Strict Cohort Disjointness:** $\text{Dev} \cap \text{Test} = \emptyset$ (0 student overlap verified).
- **Frozen Preprocessing Bounds:** All vocabularies, normalizers, and candidate resource pools are fitted exclusively on the Development cohort.
- **Training-Only SMOTE:** Resampling applied exclusively to `train.csv`, leaving validation and test sets in their natural class distribution.
- **Retraining Invariant:** Retrained corrected models are instantiated as fresh objects with distinct `ModelTrainer` instances and new optimizers.
- **Genuine Knowledge State:** Concept mastery is derived from model output probabilities, cross-verified with empirical baselines.
- **Authentic Recommendations:** Interventions recommend genuine EdNet lectures (`lectures.csv`), explanations, and questions targeting weak concepts.

---

## 5. Execution Commands

```powershell
# Activate virtual environment
& "A:\edge download\Smart_Education_Project\.venv\Scripts\Activate.ps1"

# Run data preparation and leakage audit
python scripts/prepare_data.py

# Run individual models
python scripts/run_model1.py    # LSTM + Attention
python scripts/run_model2.py    # Transformer + KT
python scripts/run_model3.py    # BERT-style + NCF
python scripts/run_model4.py    # Autoencoder + Recommender
python scripts/run_model5.py    # CNN + LSTM

# Or run complete end-to-end master pipeline (Smoke test + All models + Reports + Checklist)
python scripts/run_all.py
```
