# Smart Education Project — Complete Lab Setup & Portability Guide
=============================================================================
**Target Platform:** Windows 10 / 11 (64-bit) with NVIDIA GPU (CUDA 11.8 / 12.x)  
**Repository:** `https://github.com/anant-am/Smart-Education-Personalized-Learning.git`  
**Branch:** `main`

This guide explains how to set up, verify, and run the Smart Education AI system on a fresh lab PC from scratch.

---

## Table of Contents
- [A. Clone Repository](#a-clone-repository)
- [B. Create Python Virtual Environment](#b-create-python-virtual-environment)
- [C. Install Dependencies](#c-install-dependencies)
- [D. Environment Variable Setup (.env)](#d-environment-variable-setup-env)
- [E. Dataset Download & Directory Structure](#e-dataset-download--directory-structure)
- [F. Hardware & Environment Check](#f-hardware--environment-check)
- [G. Project Integrity Verification](#g-project-integrity-verification)
- [H. Step 1: Data Preparation & User Partitioning](#h-step-1-data-preparation--user-partitioning)
- [I. Step 2: Leakage Audit Verification](#i-step-2-leakage-audit-verification)
- [J. Step 3: Smoke-Test Pipeline Verification](#j-step-3-smoke-test-pipeline-verification)
- [K. Step 4: Individual Model Training](#k-step-4-individual-model-training)
- [L. Step 5: Full 40-Configuration Matrix Training](#l-step-5-full-40-configuration-matrix-training)
- [M. How to Resume Interrupted Training](#m-how-to-resume-interrupted-training)
- [N. How to Run the Streamlit GUI](#n-how-to-run-the-streamlit-gui)
- [O. How to Run the REST API](#o-how-to-run-the-rest-api)
- [P. How to Verify Video & Audio Attention](#p-how-to-verify-video--audio-attention)
- [Q. How to Configure Ollama Generative AI](#q-how-to-configure-ollama-generative-ai)
- [R. How to Generate Experiment Graphs & Academic Reports](#r-how-to-generate-experiment-graphs--academic-reports)
- [S. Files That Must NEVER Be Uploaded to GitHub](#s-files-that-must-never-be-uploaded-to-github)

---

## A. Clone Repository

Open PowerShell or Command Prompt on the lab PC and run:

```powershell
git clone https://github.com/anant-am/Smart-Education-Personalized-Learning.git
cd Smart-Education-Personalized-Learning
```

Verify your git state:
```powershell
git status
# Branch should be 'main'
```

---

## B. Create Python Virtual Environment

Python 3.10 to 3.12 is recommended (tested with Python 3.10 & 3.14).

```powershell
# Create virtual environment named .venv
python -m venv .venv

# Activate the virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

---

## C. Install Dependencies

Upgrade pip and install the required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU (recommended for full training), ensure PyTorch with CUDA is installed:
```powershell
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
If CUDA is False but your lab machine has an NVIDIA GPU, install the CUDA-enabled PyTorch build:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## D. Environment Variable Setup (.env)

Copy the configuration template:
```powershell
copy .env.example .env
```

Open `.env` in Notepad or VS Code and configure the paths to your local storage:
```ini
# Path to directory containing KT3 user CSVs (e.g., u1.csv, u2.csv)
EDNET_KT3_ROOT=D:\Datasets\EdNet-KT3\KT3

# Path to directory containing questions.csv, lectures.csv
EDNET_CONTENTS_ROOT=D:\Datasets\EdNet-Contents\contents

# Processed data output directory
PROCESSED_DATA_ROOT=data/processed

# Execution mode: 'smoke', 'dev', or 'full'
SMART_EDU_MODE=dev
```

*Note: All modules automatically read `.env` on startup. No manual Windows environment variable configuration is necessary.*

---

## E. Dataset Download & Directory Structure

The Smart Education project requires two datasets downloaded from the official EdNet repository (AIEd / Santa):

1. **EdNet-KT3 (Interaction Tracking):**
   - Download the KT3 interaction archive.
   - Extract to a local fast SSD/HDD directory (e.g., `D:\Datasets\EdNet-KT3\KT3`).
   - Expected structure:
     ```
     EDNET_KT3_ROOT/
     ├── u1.csv
     ├── u2.csv
     ├── ...
     └── u297915.csv (~298k user CSV files)
     ```
2. **EdNet-Contents (Pedagogical Metadata):**
   - Download the contents archive.
   - Extract to local storage (e.g., `D:\Datasets\EdNet-Contents\contents`).
   - Expected structure:
     ```
     EDNET_CONTENTS_ROOT/
     ├── questions.csv  (13,169 questions with question_id, bundle_id, correct_answer, part, tags)
     ├── lectures.csv   (1,021 video lectures with lecture_id, part, tags)
     └── ...
     ```

*Important:* Store datasets on local SSD/HDD storage during training. Do NOT train directly over network shares or synced cloud folders (OneDrive/Google Drive).

---

## F. Hardware & Environment Check

Before training, run the automated environment validator:

```powershell
python scripts/check_environment.py
```

This verifies:
- Python version and PyTorch CUDA compatibility
- Available CPU cores and host RAM
- Project storage capacity
- Configured dataset paths and pre-trained checkpoints

To enforce that raw datasets must exist immediately:
```powershell
python scripts/check_environment.py --require-datasets
```

---

## G. Project Integrity Verification

Verify that all source modules, attention components, Ollama fallbacks, and 20 required pre-trained checkpoints are intact:

```powershell
python scripts/verify_project_integrity.py
```
Expected output:
```
TOTAL INTEGRITY CHECKS EXECUTED: 57
PASSED: 57
FAILED: 0
RESULT: PASS — All required files, modules, checkpoints, and imports verified.
```

---

## H. Step 1: Data Preparation & User Partitioning

Run data preprocessing to reconstruct interaction sequences and generate the strict **User-Level 80:20 Split**:

```powershell
# Development cohort (1,000 users — rapid iteration)
python scripts/prepare_data.py --mode dev

# Full EdNet-KT3 dataset (~298,000 users)
python scripts/prepare_data.py --mode full
```

Key Invariants Enforced:
1. Ground truth correctness derived strictly by joining `student.user_answer == contents.correct_answer`.
2. Strict Disjoint User Split: `set(train_users) ∩ set(test_users) == ∅`.
3. Vocabularies and scalers fitted strictly on training users. Unseen test users transformed with frozen parameters.

---

## I. Step 2: Leakage Audit Verification

The data preparation pipeline automatically triggers the comprehensive 7-point leakage auditor:
- Check A: User Leakage (`train_users ∩ test_users == ∅`)
- Check B: Duplicate Interaction Leakage
- Check C: Target & Proxy Feature Leakage
- Check D: Temporal Timestamp Ordering
- Check E: Metadata Contamination
- Check F: Target Label Derivation
- Check G: Preprocessing Bound Quarantining

Auditor results are saved to `reports/leakage_report.md`. Verify with:
```powershell
python -c "import pandas as pd; from src.leakage_checks import LeakageAuditor; auditor = LeakageAuditor(); df = pd.read_csv('data/processed/train.csv'); res = auditor.run_all_checks(); print('Audit Passed:', res['all_passed'])"
```

---

## J. Step 3: Smoke-Test Pipeline Verification

Run a 25-user smoke test to verify all 5 models and data pipelines end-to-end in seconds:

```powershell
python scripts/prepare_data.py --mode smoke
python scripts/run_experiments.py --smoke
```

---

## K. Step 4: Individual Model Training

Train individual models with custom hyperparameters or augmentation:

```powershell
# Model 1: LSTM + Attention Knowledge Tracing
python scripts/run_model1.py

# Model 2: Transformer + Knowledge Tracing
python scripts/run_model2.py

# Model 3: BERT-style Transformer + Neural Collaborative Filtering
python scripts/run_model3.py

# Model 4: Autoencoder + Recommender Network
python scripts/run_model4.py

# Model 5: CNN + LSTM Temporal Knowledge Tracing
python scripts/run_model5.py
```

---

## L. Step 5: Full 40-Configuration Matrix Training

To execute the complete experimental matrix (5 Models × 2 Validation Strategies × 4 Augmentation Conditions):

```powershell
python scripts/run_experiments.py
```

This runs:
- **5 Models:** LSTM+Attn, Transformer+KT, BERT-NCF, Autoencoder, CNN+LSTM
- **2 Validation Strategies:** Train/Val split & 10-Fold GroupKFold
- **4 Augmentations:** Baseline, SMOTE, ADASYN, Tabular WGAN-GP
- Automatic fit diagnosis (Good Fit / Overfitting / Underfitting) & fresh-instance retraining
- Automatically exports Tables A through J to `reports/experiment_summary.md` and `reports/experiment_results.csv`.

---

## M. How to Resume Interrupted Training

`scripts/run_experiments.py` tracks state in `reports/experiment_manifest.json`. If training is stopped or interrupted, resume by running:

```powershell
python scripts/run_experiments.py --resume
```
Completed configurations are automatically detected and preserved without recomputation.

---

## N. How to Run the Streamlit GUI

Launch the interactive web application:

```powershell
streamlit run app.py
```
Or double-click `run_frontend.bat`.

The application opens at `http://localhost:8501`, providing:
1. Executive System Dashboard & Architecture Overview
2. Student Knowledge Tracing & Skill Radar Charts (189 TOEIC concept tags)
3. Multi-Model Consensus & Comparison (Models 1–5 simultaneous forward passes)
4. Pedagogical Remedial Recommendations (Lectures, Explanations, Questions)
5. Multimodal Attention Visualizer (Video & Audio Telemetry)
6. Data Leakage & Compliance Auditor
7. Generative AI Pedagogical Explanations (Ollama Tutor)

---

## O. How to Run the REST API

Launch the high-throughput asynchronous REST API:

```powershell
python api.py
```
Or via Uvicorn:
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at:
`http://localhost:8000/docs`

Key endpoints:
- `GET /health`: Health and hardware status
- `POST /predict/student`: Multi-model knowledge state & remedial recommendations
- `POST /predict/attention`: Video/audio attention classification
- `POST /predict/multimodal`: End-to-end cognitive + behavioral prediction

---

## P. How to Verify Video & Audio Attention

Run the attention regression test suite:

```powershell
python -m unittest tests/test_attention_regression.py
```
All 20 tests verify:
- OpenCV Haar cascade face/eye/blink feature extraction
- SciPy/FFmpeg acoustic feature extraction (MFCCs, spectral centroid, pitch F0)
- Gated cross-modal fusion network forward/backward passes
- Real MP4 processing, corrupted video tolerance, and missing audio stream fallback.

---

## Q. How to Configure Ollama Generative AI

Ollama provides conversational explanations for learning gaps and recommendations.

1. Install Ollama from `https://ollama.com`.
2. Pull the recommended compact model:
   ```powershell
   ollama pull qwen2.5:3b
   ```
3. Start the Ollama server:
   ```powershell
   ollama serve
   ```
4. In `.env`, set:
   ```ini
   OLLAMA_ENABLED=true
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=qwen2.5:3b
   ```

*Fallback Invariant:* If Ollama is not installed or unreachable, the system automatically engages the deterministic fallback generator (`DeterministicFallbackGenerator`). The entire ML pipeline and Streamlit dashboard remain 100% functional without Ollama.

---

## R. How to Generate Experiment Graphs & Academic Reports

The system provides an automated visualization and reporting engine (`src/experiment_visualizer.py`) that generates 16 standardized, publication-ready figures (both high-resolution 300 DPI PNG and vector SVG) and an academic Markdown report across 14 sections directly from actual empirical experiment logs (`reports/experiment_results.csv`).

### Generating Visualizations via CLI
To generate or refresh all 16 figures and the comprehensive report at any time:
```powershell
python scripts/generate_experiment_graphs.py
```
Or via the experiment runner utility:
```powershell
python scripts/run_experiments.py --visualize-only
```

*Note:* When running `python scripts/run_experiments.py`, the visualization suite is executed automatically upon experiment completion.

### Generated Figures Directory (`reports/figures/`)
The engine produces the following 16 comparison plots in both `.png` (300 DPI) and `.svg`:
1. `model_accuracy_comparison`: Test accuracy comparison across all 5 models.
2. `model_precision_comparison`: Precision metric comparison across all 5 models.
3. `model_recall_comparison`: Recall metric comparison across all 5 models.
4. `model_f1_comparison`: F1-Score comparison across all 5 models.
5. `model_roc_auc_comparison`: ROC-AUC curve comparisons across all models.
6. `validation_10fold_comparison`: Fold-level performance with mean ± standard deviation error bars.
7. `smote_comparison`: Baseline vs SMOTE synthetic oversampling comparison.
8. `adasyn_comparison`: Baseline vs ADASYN adaptive oversampling comparison.
9. `gan_comparison`: Baseline vs Tabular WGAN-GP augmentation comparison.
10. `augmentation_comparison`: Comprehensive comparison across all four augmentation conditions.
11. `fit_correction_comparison`: Pre-correction vs post-regularization overfitting mitigation.
12. `training_time_comparison`: Total wall-clock training duration (seconds).
13. `inference_latency_comparison`: Per-sample latency with 100ms real-time threshold line.
14. `ram_comparison`: Peak system RAM consumption (MB).
15. `gpu_vram_comparison`: Peak GPU VRAM memory utilization (MB).
16. `throughput_comparison`: Evaluated samples processed per second.

### Academic Report (`reports/comprehensive_experiment_report.md`)
Generates a structured 14-section comprehensive synthesis report:
1. Dataset Specification (EdNet-KT3 & Contents Metadata)
2. 80:20 User Partitioning Strategy
3. Data Leakage Audit & Verification Suite
4. Validation Methodology (Holdout vs 10-Fold GroupKFold)
5. Model Architecture Benchmark (Models 1–5)
6. 10-Fold Cross-Validation Uncertainty Analysis
7. SMOTE Oversampling Efficacy
8. ADASYN Adaptive Oversampling Efficacy
9. Tabular WGAN-GP Generative Augmentation Efficacy
10. Multi-Augmentation Comprehensive Comparison
11. Diagnostic Fit Assessment & Regularization Correction
12. Computational Efficiency & Hardware Profiling
13. Final Frozen Holdout Evaluation
14. Architectural Insights & Deployment Conclusions

### Viewing Visualizations in Streamlit GUI
The figures are directly integrated into the Streamlit frontend (`gui/page_evaluation.py`) under Section 7 (*"Generated Visualizations & Experiment Plots"*), tabbed into Model Metrics, 10-Fold CV, Augmentation, Fit Correction, and Hardware & Efficiency. If any plot has not been generated yet, the UI displays `NOT RUN` instead of placeholder data.

---

## S. Files That Must NEVER Be Uploaded to GitHub

The `.gitignore` file enforces that the following large or sensitive assets are NEVER committed:
1. **Raw EdNet Datasets:** `EdNet-KT3/`, `EdNet-Contents/`, `data/raw/`
2. **Huge Processed Interaction CSVs:** `data/processed/*.csv` (e.g., `train_dev.csv`, `train.csv`, `test.csv`, `question_events.csv`, `train_smote_flat.csv`, `train_gan_flat.csv`)
3. **Virtual Environments:** `.venv/`, `venv/`, `env/`
4. **Temporary Caches & Bytecode:** `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.cache/`
5. **Execution Logs:** `logs/*.log`
6. **Local Secrets & Credentials:** `.env`, `credentials.json`, `secrets.json`, `*.key`
7. **Raw Media Video/Audio Files:** `*.mp4`, `*.avi`, `*.wav`, `*.mp3`

---
*Prepared for the Smart Education Project Continuation & Lab Deployment.*
