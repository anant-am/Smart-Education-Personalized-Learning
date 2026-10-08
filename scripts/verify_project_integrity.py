"""
Smart Education Project — Repository & Project Integrity Verifier
===================================================================
Performs static structure audits and dynamic module import tests across:
1. Root application entrypoints (app.py, main.py, api.py, config.py)
2. Core deep learning architecture (src/models.py, dataset.py, trainer.py)
3. Data processing & leakage auditor (src/data_pipeline.py, leakage_checks.py)
4. Resampling & Augmentation (src/smote_utils.py, gan_augmentation.py)
5. Knowledge Tracing & Recommendations (src/knowledge_state.py, recommendation.py)
6. Multimodal Attention (attention/)
7. Ollama AI integration & fallback (ollama_ai/)
8. Training, benchmark & experiment scripts (scripts/)
9. Test suites (tests/)
10. Model checkpoints (models/checkpoints/)

Reports PASS / FAIL cleanly.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import importlib
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_file(rel_path: str) -> Tuple[bool, str]:
    p = PROJECT_ROOT / rel_path
    if p.exists() and p.is_file():
        return True, f"Found file: {rel_path} ({p.stat().st_size} bytes)"
    return False, f"MISSING file: {rel_path}"

def check_dir(rel_path: str) -> Tuple[bool, str]:
    p = PROJECT_ROOT / rel_path
    if p.exists() and p.is_dir():
        count = len(list(p.iterdir()))
        return True, f"Found directory: {rel_path} ({count} items)"
    return False, f"MISSING directory: {rel_path}"

def check_import(module_name: str, symbol_names: List[str] = None) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(module_name)
        if symbol_names:
            for s in symbol_names:
                if not hasattr(mod, s):
                    return False, f"Module '{module_name}' missing symbol '{s}'"
        return True, f"Successfully imported '{module_name}'"
    except Exception as e:
        return False, f"Failed importing '{module_name}': {e}"

def main():
    print("=" * 80)
    print("  SMART EDUCATION — FULL PROJECT INTEGRITY & PORTABILITY AUDIT")
    print("=" * 80)

    failures = []
    total_checks = 0

    # 1. Root Files
    print("\n[1] Checking Root Application & Configuration Files:")
    root_files = [
        "app.py", "main.py", "api.py", "config.py",
        "requirements.txt", "README.md", ".env.example", "LAB_SETUP.md"
    ]
    for rf in root_files:
        total_checks += 1
        ok, msg = check_file(rf)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # 2. Key Directories
    print("\n[2] Checking Required Project Subdirectories:")
    dirs = [
        "src", "attention", "ollama_ai", "scripts",
        "tests", "gui", "services", "models/checkpoints",
        "reports", "plots"
    ]
    for d in dirs:
        total_checks += 1
        ok, msg = check_dir(d)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # 3. Core Source Files (src/)
    print("\n[3] Checking Core Pipeline & Model Modules (src/):")
    src_files = [
        "src/__init__.py", "src/models.py", "src/dataset.py", "src/trainer.py",
        "src/data_pipeline.py", "src/leakage_checks.py", "src/smote_utils.py",
        "src/gan_augmentation.py", "src/knowledge_state.py", "src/recommendation.py",
        "src/cross_validation.py", "src/preprocessing.py"
    ]
    for sf in src_files:
        total_checks += 1
        ok, msg = check_file(sf)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # 4. Multimodal Attention Modules (attention/)
    print("\n[4] Checking Multimodal Attention Engine (attention/):")
    att_files = [
        "attention/__init__.py", "attention/attention_model.py", "attention/attention_config.py",
        "attention/attention_dataset.py", "attention/attention_evaluator.py", "attention/attention_inference.py",
        "attention/attention_pipeline.py", "attention/attention_preprocessing.py", "attention/attention_trainer.py",
        "attention/video_feature_extractor.py", "attention/audio_feature_extractor.py"
    ]
    for af in att_files:
        total_checks += 1
        ok, msg = check_file(af)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # 5. Ollama AI Modules (ollama_ai/)
    print("\n[5] Checking Ollama Generative AI Layer (ollama_ai/):")
    ollama_files = [
        "ollama_ai/__init__.py", "ollama_ai/ollama_config.py", "ollama_ai/ollama_client.py",
        "ollama_ai/learner_context.py", "ollama_ai/prompt_builder.py", "ollama_ai/response_generator.py",
        "ollama_ai/fallback.py", "ollama_ai/tutor.py"
    ]
    for of in ollama_files:
        total_checks += 1
        ok, msg = check_file(of)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # 6. Training & Experiment Scripts (scripts/)
    print("\n[6] Checking Orchestration & Training Scripts (scripts/):")
    script_files = [
        "scripts/run_experiments.py", "scripts/prepare_data.py", "scripts/check_environment.py",
        "scripts/verify_project_integrity.py", "scripts/verify_augmentation.py", "scripts/compare_models.py",
        "scripts/run_all.py", "scripts/run_model1.py", "scripts/run_model2.py", "scripts/run_model3.py",
        "scripts/run_model4.py", "scripts/run_model5.py"
    ]
    for scf in script_files:
        total_checks += 1
        ok, msg = check_file(scf)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # 7. Model Checkpoints Existence
    print("\n[7] Checking Required Pre-trained Checkpoints (models/checkpoints/):")
    required_checkpoints = [
        "models/checkpoints/model1_lstm_initial_best.pth",
        "models/checkpoints/model1_lstm_corrected_best.pth",
        "models/checkpoints/model2_transformer_initial_best.pth",
        "models/checkpoints/model2_transformer_corrected_best.pth",
        "models/checkpoints/model3_bert_ncf_initial_best.pth",
        "models/checkpoints/model3_bert_ncf_corrected_best.pth",
        "models/checkpoints/model4_autoencoder_initial_best.pth",
        "models/checkpoints/model4_autoencoder_corrected_best.pth",
        "models/checkpoints/model5_cnn_lstm_initial_best.pth",
        "models/checkpoints/model5_cnn_lstm_corrected_best.pth",
        "models/checkpoints/tabular_gan.pt",
        "models/checkpoints/attention/attention_multimodal_initial_best.pth",
        "models/checkpoints/attention/attention_multimodal_corrected_best.pth",
        "models/checkpoints/attention/attention_preprocessor.pkl",
    ]
    for ckpt in required_checkpoints:
        total_checks += 1
        ok, msg = check_file(ckpt)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # 8. Dynamic Import Verification
    print("\n[8] Dynamic Module Import & Instantiation Tests:")
    imports_to_test = [
        ("config", ["PROJECT_ROOT", "CHECKPOINTS_DIR", "PROCESSED_DATA_DIR"]),
        ("src.models", ["LSTMAttentionKT", "TransformerKT", "BERTNCF", "AutoencoderRecommender", "CNNLSTM"]),
        ("src.knowledge_state", ["KnowledgeStateEstimator"]),
        ("src.recommendation", ["RecommendationEngine"]),
        ("src.data_pipeline", ["EdNetDataPipeline"]),
        ("src.smote_utils", ["AcademicSMOTEHandler"]),
        ("src.gan_augmentation", ["TabularWGANGP", "TabularGenerator"]),
        ("attention.attention_model", ["MultimodalAttentionModel"]),
        ("attention.attention_inference", ["AttentionInferenceEngine"]),
        ("ollama_ai.ollama_client", ["OllamaClient"]),
        ("ollama_ai.fallback", ["fallback_explain_learning_gaps", "fallback_explain_recommendations"]),
    ]
    for mod_name, syms in imports_to_test:
        total_checks += 1
        ok, msg = check_import(mod_name, syms)
        print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        if not ok:
            failures.append(msg)

    # Final Verdict
    print("\n" + "=" * 80)
    print(f"  TOTAL INTEGRITY CHECKS EXECUTED: {total_checks}")
    print(f"  PASSED: {total_checks - len(failures)}")
    print(f"  FAILED: {len(failures)}")
    print("=" * 80)

    if not failures:
        print("  RESULT: PASS — All required files, modules, checkpoints, and imports verified.")
        print("=" * 80)
        sys.exit(0)
    else:
        print("  RESULT: FAIL — The following integrity checks failed:")
        for f in failures:
            print(f"    - {f}")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()
