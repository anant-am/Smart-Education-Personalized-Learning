# Smart Education Project — Repository Archival & Asset Classification Manifest
================================================================================
**Generated Date:** 2026-10-27  
**Repository:** `Smart-Education-Personalized-Learning`  
**Archival Policy:** Conservative, Non-Destructive, 100% Reversible  

---

## 1. Classification Taxonomy

To ensure zero risk of accidental loss of irreplaceable trained weights, dataset records, or research evidence, repository assets are rigorously partitioned into four distinct tiers:

### Tier 1: `KEEP-CORE` (Active Runtime & Core Architecture)
- **Trained Model Checkpoints:** All `.pth`, `.pt`, and `.pkl` weights in `models/checkpoints/` and `models/checkpoints/attention/`.
- **Preprocessed Catalogs & Vocabulary:** `data/processed/metadata.pkl`, `data/processed/questions_contents.csv`, `data/processed/lectures_contents.csv`, `data/processed/sample_test_students.csv`, `data/processed/split_users.json`.
- **Model Architecture Definitions:** `src/models.py`, `src/dataset.py`, `src/data_pipeline.py`, `src/knowledge_state.py`, `src/recommendation.py`.
- **Attention Modules:** `attention/attention_model.py`, `attention/attention_pipeline.py`, `attention/attention_inference.py`, `attention/video_feature_extractor.py`, `attention/audio_feature_extractor.py`.
- **Inference & Experiment Services:** `services/inference_service.py`, `services/experiment_service.py`, `services/__init__.py`.
- **Frontend Dashboard:** `app.py`, `gui/*.py`.
- **Generative AI Tutor:** `ollama_ai/*.py`.
- **Runtime Dependencies & Entrypoints:** `requirements-inference.txt`, `requirements-training.txt`, `config.py`, `run_frontend.bat`.

### Tier 2: `KEEP-EVIDENCE` (Empirical Benchmarks & Historical Provenance)
- **Canonical Experiment Source:** `reports/canonical_results.csv`, `reports/canonical_results_manifest.json`.
- **Raw Experiment Matrix Logs:** `reports/experiment_results.csv`, `reports/experiment_results.json`, `reports/experiment_manifest.json`.
- **Cross-Validation Records:** `reports/*_cv_10fold.csv`.
- **Standalone Benchmark Summaries:** `reports/five_model_comparison.csv`, `reports/final_comparison.csv`, `reports/augmentation_comparison.csv`.
- **Diagnostic Curves & Plots:** `reports/figures/*`, `plots/*`.
- **Research Reports:** `reports/*.md`.
- **Protected Legacy Storage:** `imp ai hackathon` (untouched, read-only snapshot).

### Tier 3: `ARCHIVE` (Stale Outputs, Redundant Drafts)
Any files identified during maintenance as non-runtime artifacts:
- No active source files or checkpoints have been deleted.
- All 204 critical baseline files are preserved with matching SHA-256 hashes in `recovery_backup_20261027_124740`.

### Tier 4: `ASK` (Uncertain Artifacts)
- No ambiguous files exist in the active tree that require user confirmation for deletion. All files are categorized under `KEEP-CORE` or `KEEP-EVIDENCE`.

---

## 2. Archival Actions Summary

- **Total Files Permanently Deleted:** 0 (Zero deletions performed)
- **Total Checkpoints Modified or Overwritten:** 0 (All weights match original SHA-256 baseline)
- **Reversibility Guarantee:** Any item staged into `archive/` can be restored immediately without data loss.
