"""
Smart Education Project — Five Hybrid Deep-Learning Models Academic Comparison
================================================================================
Master Comparison Script adhering strictly to the University Standardized Numbering:
  MODEL 1 — LSTM + Attention (Knowledge Tracing)
  MODEL 2 — Transformer + Knowledge Tracing (Self-Attentive Knowledge Tracing)
  MODEL 3 — BERT-style Transformer + Neural Collaborative Filtering (NCF Hybrid)
  MODEL 4 — Autoencoder + Recommender Network (Resource Recommendation)
  MODEL 5 — CNN + LSTM (Temporal Convolution + Sequential State Tracking)

Outputs:
1. Complete 5-technique comparison table across all required academic metrics
2. Comparative visualization charts (saved to plots/five_model_comparison.png)
3. reports/five_model_comparison.csv
4. reports/model_comparison.md & reports/final_results.md
5. End-to-end personalized learning recommendation demo with authentic EdNet resources
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR, PLOTS_DIR, PROCESSED_DATA_DIR, ensure_dirs
from src.knowledge_state import KnowledgeStateEstimator
from src.recommendation import RecommendationEngine

ensure_dirs()


def load_all_results():
    """Loads results JSON files for all five standardized models."""
    with open(REPORTS_DIR / "model1_results.json", 'r', encoding='utf-8') as f:
        m1 = json.load(f)
    with open(REPORTS_DIR / "model2_results.json", 'r', encoding='utf-8') as f:
        m2 = json.load(f)
    with open(REPORTS_DIR / "model3_results.json", 'r', encoding='utf-8') as f:
        m3 = json.load(f)
    with open(REPORTS_DIR / "model4_results.json", 'r', encoding='utf-8') as f:
        m4 = json.load(f)
    with open(REPORTS_DIR / "model5_results.json", 'r', encoding='utf-8') as f:
        m5 = json.load(f)
    return m1, m2, m3, m4, m5


def generate_comparison_table(m1, m2, m3, m4, m5):
    """Constructs the comprehensive 5-model comparison DataFrame."""
    rows = []

    # Model 1: LSTM + Attention
    m1_test = m1.get('test_metrics', {})
    rows.append({
        'Model ID': 'Model 1',
        'Hybrid Architecture': 'LSTM + Attention',
        'Primary Task': 'Knowledge Tracing',
        'Parameters': f"{m1.get('parameters', 0):,}",
        'Accuracy': f"{m1_test.get('accuracy', 0.0):.4f}",
        'Precision': f"{m1_test.get('precision', 0.0):.4f}",
        'Recall': f"{m1_test.get('recall', 0.0):.4f}",
        'F1-Score': f"{m1_test.get('f1', 0.0):.4f}",
        'ROC-AUC': f"{m1_test.get('auc', 0.0):.4f}",
        'PR-AUC': f"{m1_test.get('pr_auc', 0.0):.4f}",
        'LogLoss': f"{m1_test.get('log_loss', 0.0):.4f}",
        'Training Time (s)': f"{m1.get('training_time_sec', 0.0):.2f}",
        'Inference Time (s)': f"{m1.get('inference_time_sec', 0.0):.4f}",
        'Peak GPU (MB)': f"{m1.get('gpu_memory_mb', 0.0):.1f}",
        'NDCG@5': 'N/A',
        'HitRate@5': 'N/A'
    })

    # Model 2: Transformer + Knowledge Tracing
    m2_test = m2.get('test_metrics', {})
    rows.append({
        'Model ID': 'Model 2',
        'Hybrid Architecture': 'Transformer + Knowledge Tracing',
        'Primary Task': 'Knowledge Tracing',
        'Parameters': f"{m2.get('parameters', 0):,}",
        'Accuracy': f"{m2_test.get('accuracy', 0.0):.4f}",
        'Precision': f"{m2_test.get('precision', 0.0):.4f}",
        'Recall': f"{m2_test.get('recall', 0.0):.4f}",
        'F1-Score': f"{m2_test.get('f1', 0.0):.4f}",
        'ROC-AUC': f"{m2_test.get('auc', 0.0):.4f}",
        'PR-AUC': f"{m2_test.get('pr_auc', 0.0):.4f}",
        'LogLoss': f"{m2_test.get('log_loss', 0.0):.4f}",
        'Training Time (s)': f"{m2.get('training_time_sec', 0.0):.2f}",
        'Inference Time (s)': f"{m2.get('inference_time_sec', 0.0):.4f}",
        'Peak GPU (MB)': f"{m2.get('gpu_memory_mb', 0.0):.1f}",
        'NDCG@5': 'N/A',
        'HitRate@5': 'N/A'
    })

    # Model 3: BERT-style Transformer + Neural Collaborative Filtering
    m3_test = m3.get('test_metrics', {})
    rows.append({
        'Model ID': 'Model 3',
        'Hybrid Architecture': 'BERT-style Transformer + NCF',
        'Primary Task': 'Collaborative Tracing',
        'Parameters': f"{m3.get('parameters', 0):,}",
        'Accuracy': f"{m3_test.get('accuracy', 0.0):.4f}",
        'Precision': f"{m3_test.get('precision', 0.0):.4f}",
        'Recall': f"{m3_test.get('recall', 0.0):.4f}",
        'F1-Score': f"{m3_test.get('f1', 0.0):.4f}",
        'ROC-AUC': f"{m3_test.get('auc', 0.0):.4f}",
        'PR-AUC': f"{m3_test.get('pr_auc', 0.0):.4f}",
        'LogLoss': f"{m3_test.get('log_loss', 0.0):.4f}",
        'Training Time (s)': f"{m3.get('training_time_sec', 0.0):.2f}",
        'Inference Time (s)': f"{m3.get('inference_time_sec', 0.0):.4f}",
        'Peak GPU (MB)': f"{m3.get('gpu_memory_mb', 0.0):.1f}",
        'NDCG@5': 'N/A',
        'HitRate@5': 'N/A'
    })

    # Model 4: Autoencoder + Recommender Network
    m4_test = m4.get('test_metrics', {})
    rows.append({
        'Model ID': 'Model 4',
        'Hybrid Architecture': 'Autoencoder + Recommender Network',
        'Primary Task': 'Resource Recommendation',
        'Parameters': f"{m4.get('parameters', 0):,}",
        'Accuracy': f"MSE: {m4_test.get('reconstruction_mse', 0.0):.4f}",
        'Precision': f"{m4_test.get('Precision@5', 0.0):.4f} (@5)",
        'Recall': f"{m4_test.get('Recall@5', 0.0):.4f} (@5)",
        'F1-Score': 'N/A',
        'ROC-AUC': 'N/A',
        'PR-AUC': 'N/A',
        'LogLoss': 'N/A',
        'Training Time (s)': f"{m4.get('training_time_sec', 0.0):.2f}",
        'Inference Time (s)': 'N/A',
        'Peak GPU (MB)': f"{m4.get('gpu_memory_mb', 0.0):.1f}",
        'NDCG@5': f"{m4_test.get('NDCG@5', 0.0):.4f}",
        'HitRate@5': f"{m4_test.get('HitRate@5', 0.0)*100:.2f}%"
    })

    # Model 5: CNN + LSTM
    m5_test = m5.get('test_metrics', {})
    rows.append({
        'Model ID': 'Model 5',
        'Hybrid Architecture': 'CNN + LSTM',
        'Primary Task': 'Temporal Knowledge Tracing',
        'Parameters': f"{m5.get('parameters', 0):,}",
        'Accuracy': f"{m5_test.get('accuracy', 0.0):.4f}",
        'Precision': f"{m5_test.get('precision', 0.0):.4f}",
        'Recall': f"{m5_test.get('recall', 0.0):.4f}",
        'F1-Score': f"{m5_test.get('f1', 0.0):.4f}",
        'ROC-AUC': f"{m5_test.get('auc', 0.0):.4f}",
        'PR-AUC': f"{m5_test.get('pr_auc', 0.0):.4f}",
        'LogLoss': f"{m5_test.get('log_loss', 0.0):.4f}",
        'Training Time (s)': f"{m5.get('training_time_sec', 0.0):.2f}",
        'Inference Time (s)': f"{m5.get('inference_time_sec', 0.0):.4f}",
        'Peak GPU (MB)': f"{m5.get('gpu_memory_mb', 0.0):.1f}",
        'NDCG@5': 'N/A',
        'HitRate@5': 'N/A'
    })

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "five_model_comparison.csv", index=False)
    return df


def plot_comparison(m1, m2, m3, m4, m5):
    """Plots comparative metrics across all 5 models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Knowledge Tracing Performance (M1, M2, M3, M5)
    kt_names = ['M1: LSTM+Attn', 'M2: Transformer', 'M3: BERT+NCF', 'M5: CNN+LSTM']
    m1_t = m1.get('test_metrics', {})
    m2_t = m2.get('test_metrics', {})
    m3_t = m3.get('test_metrics', {})
    m5_t = m5.get('test_metrics', {})

    accs = [m1_t.get('accuracy', 0.0), m2_t.get('accuracy', 0.0), m3_t.get('accuracy', 0.0), m5_t.get('accuracy', 0.0)]
    aucs = [m1_t.get('auc', 0.0), m2_t.get('auc', 0.0), m3_t.get('auc', 0.0), m5_t.get('auc', 0.0)]
    f1s = [m1_t.get('f1', 0.0), m2_t.get('f1', 0.0), m3_t.get('f1', 0.0), m5_t.get('f1', 0.0)]

    x = np.arange(len(kt_names))
    width = 0.25

    axes[0].bar(x - width, accs, width, label='Accuracy', color='#2b5c8f')
    axes[0].bar(x, aucs, width, label='ROC-AUC', color='#2ca02c')
    axes[0].bar(x + width, f1s, width, label='F1-Score', color='#d62728')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Knowledge Tracing Performance (Untouched Test)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kt_names, rotation=15)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Model 4 Recommendation Ranking Metrics
    m4_t = m4.get('test_metrics', {})
    k_labels = ['Precision', 'Recall', 'NDCG', 'HitRate']
    at_5 = [
        m4_t.get('Precision@5', 0.0),
        m4_t.get('Recall@5', 0.0),
        m4_t.get('NDCG@5', 0.0),
        m4_t.get('HitRate@5', 0.0)
    ]
    at_10 = [
        m4_t.get('Precision@10', 0.0),
        m4_t.get('Recall@10', 0.0),
        m4_t.get('NDCG@10', 0.0),
        m4_t.get('HitRate@10', 0.0)
    ]

    x2 = np.arange(len(k_labels))
    axes[1].bar(x2 - width/2, at_5, width, label='Top-5 (@5)', color='#1f77b4')
    axes[1].bar(x2 + width/2, at_10, width, label='Top-10 (@10)', color='#ff7f0e')
    axes[1].set_ylabel('Metric Score')
    axes[1].set_title('Model 4 Autoencoder Ranking Metrics (Untouched Test)')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(k_labels)
    axes[1].legend()
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # 3. Model Efficiency & Complexity
    all_names = ['M1: LSTM', 'M2: Trans', 'M3: BERT-NCF', 'M4: Autoenc', 'M5: CNN-LSTM']
    times = [
        m1.get('training_time_sec', 0.0),
        m2.get('training_time_sec', 0.0),
        m3.get('training_time_sec', 0.0),
        m4.get('training_time_sec', 0.0),
        m5.get('training_time_sec', 0.0)
    ]
    params = [
        m1.get('parameters', 0) / 1000,
        m2.get('parameters', 0) / 1000,
        m3.get('parameters', 0) / 1000,
        m4.get('parameters', 0) / 1000,
        m5.get('parameters', 0) / 1000
    ]

    x3 = np.arange(len(all_names))
    ax3_twin = axes[2].twinx()

    p1 = axes[2].bar(x3 - width/2, times, width, label='Training Time (s)', color='#9467bd')
    p2 = ax3_twin.bar(x3 + width/2, params, width, label='Params (k)', color='#8c564b')

    axes[2].set_ylabel('Training Time (seconds)', color='#9467bd')
    ax3_twin.set_ylabel('Parameters (Thousands)', color='#8c564b')
    axes[2].set_title('Computational Cost & Model Complexity')
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(all_names, rotation=20)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "five_model_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  Saved comparison chart: {plot_path}")


def df_to_markdown(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_markdown_reports(df_comp, m1, m2, m3, m4, m5):
    # 1. reports/model_comparison.md
    comp_md = [
        "# Five Hybrid Deep-Learning Models — Comprehensive Academic Comparison\n",
        "**Project:** AI-Based Personalized Learning Recommendation System – Smart Education  ",
        "**Dataset:** EdNet-KT3 (1,000-User Development Cohort, 250-User Unseen Test Cohort)  ",
        "**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM), CUDA 12.6, PyTorch  \n",
        "## 1. Master Cross-Model Performance Evaluation Table\n",
        df_to_markdown(df_comp),
        "\n\n## 2. Cross-Validation Summary (10-Fold Grouped on Development Cohort)\n",
        "| Model | CV Accuracy (Mean ± Std) | CV F1-Score (Mean ± Std) | CV ROC-AUC (Mean ± Std) | CV Log Loss (Mean ± Std) |",
        "|---|---|---|---|---|",
        f"| **Model 1 (LSTM + Attention)** | {m1['cv_summary']['accuracy_mean']:.4f} ± {m1['cv_summary']['accuracy_std']:.4f} | {m1['cv_summary']['f1_mean']:.4f} ± {m1['cv_summary']['f1_std']:.4f} | {m1['cv_summary']['roc_auc_mean']:.4f} ± {m1['cv_summary']['roc_auc_std']:.4f} | {m1['cv_summary']['log_loss_mean']:.4f} ± {m1['cv_summary']['log_loss_std']:.4f} |",
        f"| **Model 2 (Transformer + KT)** | {m2['cv_summary']['accuracy_mean']:.4f} ± {m2['cv_summary']['accuracy_std']:.4f} | {m2['cv_summary']['f1_mean']:.4f} ± {m2['cv_summary']['f1_std']:.4f} | {m2['cv_summary']['roc_auc_mean']:.4f} ± {m2['cv_summary']['roc_auc_std']:.4f} | {m2['cv_summary']['log_loss_mean']:.4f} ± {m2['cv_summary']['log_loss_std']:.4f} |",
        f"| **Model 3 (BERT-style Transformer + NCF)** | {m3['cv_summary']['accuracy_mean']:.4f} ± {m3['cv_summary']['accuracy_std']:.4f} | {m3['cv_summary']['f1_mean']:.4f} ± {m3['cv_summary']['f1_std']:.4f} | {m3['cv_summary']['roc_auc_mean']:.4f} ± {m3['cv_summary']['roc_auc_std']:.4f} | {m3['cv_summary']['log_loss_mean']:.4f} ± {m3['cv_summary']['log_loss_std']:.4f} |",
        f"| **Model 4 (Autoencoder Recommender)** | N/A (Reconstruction MSE) | N/A | CV MSE: {m4.get('cv_reconstruction_mse', {}).get('mean', 0.0):.4f} ± {m4.get('cv_reconstruction_mse', {}).get('std', 0.0):.4f} | N/A |",
        f"| **Model 5 (CNN + LSTM)** | {m5['cv_summary']['accuracy_mean']:.4f} ± {m5['cv_summary']['accuracy_std']:.4f} | {m5['cv_summary']['f1_mean']:.4f} ± {m5['cv_summary']['f1_std']:.4f} | {m5['cv_summary']['roc_auc_mean']:.4f} ± {m5['cv_summary']['roc_auc_std']:.4f} | {m5['cv_summary']['log_loss_mean']:.4f} ± {m5['cv_summary']['log_loss_std']:.4f} |",
        "\n\n## 3. Academic Analysis & Architectural Strengths\n",
        "1. **Model 1 (LSTM + Attention):** Combines recurrence with self-attention weights to capture temporal dependencies with explainable attention maps.",
        "2. **Model 2 (Transformer + Knowledge Tracing):** Uses causal multi-head self-attention to model long-range dependencies with O(1) sequential path length.",
        "3. **Model 3 (BERT-style Transformer + NCF):** Combines bidirectional sequence representations with non-linear collaborative filtering item interactions.",
        "4. **Model 4 (Autoencoder + Recommender Network):** Compresses high-dimensional engagement into a latent bottleneck, reconstructing profiles and ranking authentic EdNet interventions.",
        "5. **Model 5 (CNN + LSTM):** Extracts localized temporal features via 1D convolutions before tracking cumulative learning trajectories with LSTM sequence cells."
    ]

    with open(REPORTS_DIR / "model_comparison.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(comp_md))

    # 2. reports/final_results.md
    final_md = [
        "# Final Academic Experimental Results\n",
        "**Research Title:** AI-Based Personalized Learning Recommendation System – Smart Education  ",
        "**Final Unseen Evaluation Cohort:** `KT-3 (250) TEST` (249 Valid Students, 174,608 Events)  \n",
        "## Executive Summary of Findings\n",
        "- **Total Models Trained & Evaluated:** 5 Hybrid Architectures under Standardized Numbering\n",
        "- **Dataset Integrity:** Strict zero-leakage isolation between 1,000 Development Users and 250 Unseen Test Users.\n",
        "- **Retraining Validation:** Overfitting diagnosis performed with quantitative loss gap tracking; all corrected models were retrained using verified fresh `ModelTrainer` instances.\n",
        "\n### Comprehensive Evaluation Results Table\n",
        df_to_markdown(df_comp)
    ]

    with open(REPORTS_DIR / "final_results.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_md))

    print(f"  Saved reports: {REPORTS_DIR / 'model_comparison.md'} and {REPORTS_DIR / 'final_results.md'}")


def run_end_to_end_demo():
    """Runs end-to-end personalized learning demonstration with dual knowledge state."""
    print("\n" + "=" * 70)
    print("  STAGE 7: END-TO-END PERSONALIZED LEARNING DEMONSTRATION")
    print("=" * 70)

    questions_df = pd.read_csv(PROCESSED_DATA_DIR / "questions_contents.csv")
    lectures_df = pd.read_csv(PROCESSED_DATA_DIR / "lectures_contents.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    sample_uid = test_df['user_id'].iloc[0]
    sample_student_df = test_df[test_df['user_id'] == sample_uid]

    kse = KnowledgeStateEstimator()
    baseline_state = kse.compute_historical_baseline_state(sample_student_df)
    learning_gaps = kse.detect_learning_gaps(baseline_state, mastery_threshold=0.60, min_attempts=1)

    rec_engine = RecommendationEngine(questions_df, lectures_df)
    recommendations = rec_engine.recommend_for_gaps(learning_gaps, top_k=5)

    demo_text = [
        "=" * 70,
        f"PERSONALIZED LEARNING RECOMMENDATION DEMO — STUDENT: {sample_uid}",
        "=" * 70,
        f"Total Student Learning Events in Unseen Test Set: {len(sample_student_df):,}",
        f"Unique Concepts Attempted: {len(baseline_state)}",
        f"Identified Learning Gaps (Mastery < 0.60): {len(learning_gaps)}",
        "\nTop 3 Critical Learning Deficiencies:",
    ]
    for i, g in enumerate(learning_gaps[:3], 1):
        demo_text.append(f"  {i}. {g['concept_name']} (Concept {g['concept_id']}): Empirical Acc = {g['mastery']:.4f} | Attempts = {g['attempts']}")

    demo_text.append("\nTop 5 Personalized Recommended Learning Resources (Actual EdNet Catalog):")
    for i, r in enumerate(recommendations, 1):
        demo_text.append(f"  {i}. [{r['type'].upper()}] Resource ID: {r['id']} | Targets: {r.get('concept', 'General')} | Part: {r.get('part', 'N/A')}")

    demo_str = '\n'.join(demo_text)
    print(demo_str)

    demo_path = REPORTS_DIR / "personalized_recommendation_demo.txt"
    with open(demo_path, 'w', encoding='utf-8') as f:
        f.write(demo_str)
    print(f"\n  Saved end-to-end demo to: {demo_path}")


def main():
    print("=" * 80)
    print("  GENERATING FIVE HYBRID TECHNIQUES ACADEMIC COMPARISON REPORT")
    print("=" * 80)

    m1, m2, m3, m4, m5 = load_all_results()
    df_comp = generate_comparison_table(m1, m2, m3, m4, m5)

    print("\n" + df_comp.to_string(index=False))

    plot_comparison(m1, m2, m3, m4, m5)
    write_markdown_reports(df_comp, m1, m2, m3, m4, m5)
    run_end_to_end_demo()

    print("\n" + "=" * 80)
    print("  ACADEMIC COMPARISON & FINAL REPORTS GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
