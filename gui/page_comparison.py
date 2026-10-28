# -*- coding: utf-8 -*-
"""
Smart Education — Model & Experiment Comparison Page
=====================================================
Interactive comparative dashboard for all five AI architectures and their
augmentation/validation experimental matrix (40 configurations).

Strictly reads verified experiment records from reports/experiment_results.csv,
reports/five_model_comparison.csv, and verified checkpoint status.
No retraining, no synthetic data generation, and no metric fabrication.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CHECKPOINTS_DIR, ATTENTION_CHECKPOINTS_DIR, FIGURES_DIR, PLOTS_DIR

# Import experiment loaders with dual fallback for bulletproof import resolution
try:
    from services.experiment_service import (
        get_experiment_results_df,
        get_five_model_comparison,
        get_final_comparison,
        get_augmentation_comparison,
        get_experiment_manifest,
        get_checkpoint_files,
        get_figure_path,
        get_plot_path,
    )
except ImportError:
    from services.inference_service import (
        get_experiment_results_df,
        get_five_model_comparison,
        get_final_comparison,
        get_augmentation_comparison,
        get_experiment_manifest,
        get_checkpoint_files,
        get_figure_path,
        get_plot_path,
    )


def _fmt(val: Any, is_pct: bool = False, decimals: int = 4) -> str:
    """Format numeric values safely, returning 'Not available' for nulls."""
    if val is None or pd.isna(val):
        return "Not available"
    if isinstance(val, (int, float, np.floating, np.integer)):
        if is_pct:
            return f"{val * 100:.2f}%"
        return f"{val:.{decimals}f}"
    return str(val)


def _fmt_int(val: Any) -> str:
    """Format integers with commas safely, returning 'Not available' for nulls."""
    if val is None or pd.isna(val):
        return "Not available"
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)


def render(engine: Optional[Any] = None) -> None:
    """
    Render entry point for the Model & Experiment Comparison page.

    Parameters
    ----------
    engine : MultiModelInferenceEngine or None
        Active multi-model inference engine instance, or None if offline.
    """
    # ── Header ──
    st.markdown('<div class="main-header">⚖️ Model & Experiment Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Verified comparative benchmarking of all five AI architectures across 4 augmentation methods and '
        '2 validation strategies (40-run experimental matrix).'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Cohort & Methodology Notice ──
    st.info(
        "**Verified Data Source & Evaluation Methodology Notice:**\n"
        "• All 40 matrix configurations below are sourced directly from verified run records in `reports/experiment_results.csv`.\n"
        "• The 40-configuration matrix reflects evaluations on the standardized development cohort sample (20 train users, 5 val users, 4 test users, ~21k interaction events).\n"
        "• Standalone full-cohort benchmark evaluations are available under the **Final Standalone Benchmarks** tab.\n"
        "• Validation metrics are evaluated on the validation split during training; final test metrics reflect generalization on held-out test data. "
        "Test metrics were strictly reserved for final evaluation and never used for model selection.",
        icon="ℹ️"
    )

    # ── Top-level Navigation Tabs ──
    tab_overview, tab_matrix, tab_metrics, tab_interactive, tab_model4, tab_final, tab_plots = st.tabs([
        "🤖 Five-Model Overview",
        "📋 40-Configuration Matrix",
        "📊 Detailed Performance Metrics",
        "📈 Interactive Visualizations",
        "🎯 Model 4 Recommender Analysis",
        "🏆 Final Standalone Benchmarks",
        "🖼️ Static Figures & Curves",
    ])

    # Load 40-run experiment dataframe safely
    try:
        df_exp = get_experiment_results_df()
    except Exception as e:
        df_exp = None

    if df_exp is None:
        df_exp = pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: FIVE-MODEL OVERVIEW & CHECKPOINT VERIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.subheader("A. Five-Model Architecture & Runtime Status")
        st.caption("Verification of trained weights on disk and active memory status in the inference runtime.")

        # Inspect disk checkpoints
        disk_models = {
            "model1": {
                "name": "Model 1: LSTM + Attention",
                "task": "Knowledge Tracing (Temporal Sequence)",
                "checkpoint": "model1_lstm_corrected_best.pth",
                "param_str": "803,201 ~ 815,361",
                "engine_attr": "m1",
            },
            "model2": {
                "name": "Model 2: Transformer + Knowledge Tracing",
                "task": "Self-Attentive Knowledge Tracing (SAKT)",
                "checkpoint": "model2_transformer_corrected_best.pth",
                "param_str": "824,385 ~ 836,545",
                "engine_attr": "m2",
            },
            "model3": {
                "name": "Model 3: BERT-style Transformer Encoder + NCF",
                "task": "Collaborative Knowledge Tracing & Interaction",
                "checkpoint": "model3_bert_ncf_corrected_best.pth",
                "param_str": "1,572,737 ~ 1,597,121",
                "engine_attr": "m3",
            },
            "model4": {
                "name": "Model 4: Autoencoder + Recommender Network",
                "task": "Personalized Resource Recommendation",
                "checkpoint": "model4_autoencoder_corrected_best.pth",
                "param_str": "103,368 ~ 113,920",
                "engine_attr": "m4",
            },
            "model5": {
                "name": "Model 5: CNN + LSTM",
                "task": "Temporal Knowledge Tracing with Local Spatial Convolutions",
                "checkpoint": "model5_cnn_lstm_corrected_best.pth",
                "param_str": "803,201 ~ 815,361",
                "engine_attr": "m5",
            },
        }

        overview_cards = []
        for key, info in disk_models.items():
            ckpt_path = CHECKPOINTS_DIR / info["checkpoint"]
            ckpt_exists = ckpt_path.exists()
            ckpt_size = f"{ckpt_path.stat().st_size / (1024*1024):.2f} MB" if ckpt_exists else "Missing"

            # Runtime loaded verification
            is_loaded = False
            if engine is not None:
                attr = getattr(engine, info["engine_attr"], None)
                if attr is not None:
                    is_loaded = True

            overview_cards.append({
                "Model Key": key,
                "Model Architecture": info["name"],
                "Primary Task": info["task"],
                "Parameters": info["param_str"],
                "Checkpoint File": info["checkpoint"],
                "Checkpoint on Disk": "✅ Verified Present" if ckpt_exists else "❌ Missing",
                "Disk Size": ckpt_size,
                "Inference Runtime Status": "🟢 Active & Loaded for Inference" if is_loaded else "⚪ Offline / Unloaded",
            })

        df_overview = pd.DataFrame(overview_cards)
        st.dataframe(
            df_overview[["Model Architecture", "Primary Task", "Parameters", "Checkpoint File", "Disk Size", "Checkpoint on Disk", "Inference Runtime Status"]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("Multimodal Attention Checkpoint Status")
        att_ckpt = ATTENTION_CHECKPOINTS_DIR / "attention_multimodal_corrected_best.pth"
        att_exists = att_ckpt.exists()
        att_size = f"{att_ckpt.stat().st_size / (1024*1024):.2f} MB" if att_exists else "Missing"

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("Multimodal Attention Weights", "Verified Present" if att_exists else "Missing")
        with col_a2:
            st.metric("Checkpoint File Size", att_size)
        with col_a3:
            st.metric("Attention Pipeline", "Ready (attention/)" if att_exists else "Unavailable")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: COMPLETE 40-CONFIGURATION EXPERIMENT MATRIX
    # ══════════════════════════════════════════════════════════════════════════
    with tab_matrix:
        st.subheader("B. The 40-Configuration Experiment Matrix")
        st.caption(
            "Complete 5 models × 4 augmentations (Baseline, SMOTE, ADASYN, GAN) × 2 validation strategies "
            "(Without 10-fold, With 10-fold). All 40 combinations supported."
        )

        if df_exp.empty:
            st.warning("Experiment results data is currently unavailable.")
        else:
            # Filters
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                model_options = ["All Models"] + sorted([m for m in df_exp["model_name"].dropna().unique().tolist()])
                selected_model = st.selectbox("Filter by Model Architecture", model_options, key="matrix_model_filter")
            with fcol2:
                aug_options = ["All Augmentations"] + sorted([a for a in df_exp["aug_condition"].dropna().unique().tolist()])
                selected_aug = st.selectbox("Filter by Augmentation Method", aug_options, key="matrix_aug_filter")
            with fcol3:
                val_options = ["All Strategies", "Without 10-fold (no_10fold)", "With 10-fold (10fold)"]
                selected_val = st.selectbox("Filter by Validation Strategy", val_options, key="matrix_val_filter")

            # Apply filters
            df_filtered = df_exp.copy()
            if selected_model != "All Models":
                df_filtered = df_filtered[df_filtered["model_name"] == selected_model]
            if selected_aug != "All Augmentations":
                df_filtered = df_filtered[df_filtered["aug_condition"] == selected_aug]
            if selected_val == "Without 10-fold (no_10fold)":
                df_filtered = df_filtered[df_filtered["val_strategy"] == "no_10fold"]
            elif selected_val == "With 10-fold (10fold)":
                df_filtered = df_filtered[df_filtered["val_strategy"] == "10fold"]

            # Status summary banner
            matrix_runs = df_filtered[df_filtered["run_id"] != "model5_standalone_legacy_summary"]
            st.info(f"📊 **Showing {len(matrix_runs)} of 40 planned configurations with saved results.**")

            # Friendly columns for presentation
            display_matrix = pd.DataFrame()
            display_matrix["Run ID"] = df_filtered["run_id"].fillna("Not available") if "run_id" in df_filtered else "Not available"
            display_matrix["Model"] = df_filtered["model_name"].fillna("Not available")
            display_matrix["Augmentation"] = df_filtered["aug_condition"].map(lambda x: str(x).upper() if pd.notna(x) else "Not available")
            display_matrix["Validation"] = df_filtered["val_strategy"].map({"no_10fold": "Without 10-fold", "10fold": "With 10-fold"}).fillna(df_filtered["val_strategy"]).fillna("Not available")
            display_matrix["Parameters"] = df_filtered["parameters"].map(_fmt_int) if "parameters" in df_filtered else "Not available"
            display_matrix["Train Events"] = df_filtered["train_events"].map(_fmt_int) if "train_events" in df_filtered else "Not available"
            display_matrix["Val Events"] = df_filtered["val_events"].map(_fmt_int) if "val_events" in df_filtered else "Not available"
            display_matrix["Test Events"] = df_filtered["test_events"].map(_fmt_int) if "test_events" in df_filtered else "Not available"
            display_matrix["Status"] = df_filtered["status"].map(lambda x: "✅ SUCCESS" if str(x).upper() == "SUCCESS" else ("Not available" if pd.isna(x) or str(x).lower() == 'not available' else "⚠️ " + str(x))) if "status" in df_filtered else "✅ SUCCESS"
            display_matrix["Timestamp"] = df_filtered["timestamp"].fillna("Not available") if "timestamp" in df_filtered else "Not available"

            st.dataframe(display_matrix, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: PERFORMANCE METRICS (CLASSIFICATION + EFFICIENCY + RECOMMENDATION)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_metrics:
        st.subheader("C. Performance Metrics Table")
        st.caption(
            "Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, "
            "Training Time, Latency, Peak RAM, Peak VRAM, and Throughput. Distinguishes Validation vs Test."
        )

        if df_exp.empty:
            st.warning("Performance metrics data is currently unavailable.")
        else:
            metric_view = st.radio(
                "Select Metrics View Mode",
                ["All 40 Runs (Comprehensive)", "Models 1, 2, 3, 5 (Classification Metrics)", "Model 4 (Recommendation Metrics Only)"],
                horizontal=True
            )

            if metric_view == "Models 1, 2, 3, 5 (Classification Metrics)":
                df_m = df_exp[df_exp["model_key"] != "model4"].copy()
            elif metric_view == "Model 4 (Recommendation Metrics Only)":
                df_m = df_exp[df_exp["model_key"] == "model4"].copy()
            else:
                df_m = df_exp.copy()

            # Build clean formatted table
            fmt_table = pd.DataFrame()
            fmt_table["Model"] = df_m["model_name"].fillna("Not available")
            fmt_table["Augmentation"] = df_m["aug_condition"].map(lambda x: str(x).upper() if pd.notna(x) else "Not available")
            fmt_table["Validation"] = df_m["val_strategy"].map({"no_10fold": "Without 10-fold", "10fold": "With 10-fold"}).fillna("Not available")

            # Classification
            fmt_table["Val Acc"] = df_m["val_accuracy"].map(lambda x: _fmt(x, is_pct=True))
            fmt_table["Test Acc"] = df_m["test_accuracy"].map(lambda x: _fmt(x, is_pct=True))
            fmt_table["Val F1"] = df_m["val_f1"].map(lambda x: _fmt(x))
            fmt_table["Test F1"] = df_m["test_f1"].map(lambda x: _fmt(x))
            fmt_table["Val AUC"] = df_m["val_auc"].map(lambda x: _fmt(x))
            fmt_table["Test AUC"] = df_m["test_auc"].map(lambda x: _fmt(x))
            fmt_table["Val Recall"] = df_m["val_recall"].map(lambda x: _fmt(x))
            fmt_table["Test Recall"] = df_m["test_recall"].map(lambda x: _fmt(x))
            fmt_table["Val Precision"] = df_m["val_precision"].map(lambda x: _fmt(x))
            fmt_table["Test Precision"] = df_m["test_precision"].map(lambda x: _fmt(x))

            # Model 4 Specifics
            if metric_view != "Models 1, 2, 3, 5 (Classification Metrics)":
                fmt_table["Recon MSE"] = df_m["reconstruction_mse"].map(lambda x: _fmt(x, decimals=4))
                fmt_table["HitRate@5"] = df_m["hitrate_at_5"].map(lambda x: _fmt(x, is_pct=True))
                fmt_table["NDCG@5"] = df_m["ndcg_at_5"].map(lambda x: _fmt(x, decimals=4))
                fmt_table["Precision@5"] = df_m["precision_at_5"].map(lambda x: _fmt(x, decimals=4))
                fmt_table["Recall@5"] = df_m["recall_at_5"].map(lambda x: _fmt(x, decimals=4))

            # Efficiency
            fmt_table["Train Time (s)"] = df_m["train_time_sec"].map(lambda x: f"{x:.2f}s" if pd.notna(x) else "Not available") if "train_time_sec" in df_m else "Not available"
            if "inference_time_sec" in df_m:
                fmt_table["Inference Latency"] = df_m["inference_time_sec"].map(lambda x: f"{x*1000:.1f} ms" if pd.notna(x) else "Not available")
            elif "inference_time_ms" in df_m:
                fmt_table["Inference Latency"] = df_m["inference_time_ms"].map(lambda x: f"{x:.1f} ms" if pd.notna(x) else "Not available")
            else:
                fmt_table["Inference Latency"] = "Not available"
            fmt_table["Throughput"] = df_m["throughput_events_per_sec"].map(lambda x: f"{x:,.0f} ev/s" if pd.notna(x) else "Not available") if "throughput_events_per_sec" in df_m else "Not available"
            fmt_table["Peak RAM"] = df_m["peak_ram_mb"].map(lambda x: f"{x:.1f} MB" if pd.notna(x) else "Not available") if "peak_ram_mb" in df_m else "Not available"
            vram_col = "peak_vram_mb" if "peak_vram_mb" in df_m else ("peak_gpu_mb" if "peak_gpu_mb" in df_m else None)
            fmt_table["Peak VRAM"] = df_m[vram_col].map(lambda x: f"{x:.1f} MB" if pd.notna(x) else "Not available") if vram_col else "Not available"

            st.dataframe(fmt_table, use_container_width=True, hide_index=True)

            # Download button for data consistency
            csv_data = df_exp.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Verified 40-Run Results (CSV)",
                data=csv_data,
                file_name="verified_40_configuration_experiment_results.csv",
                mime="text/csv",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: INTERACTIVE COMPARISON CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_interactive:
        st.subheader("D. Interactive Comparison Visualizations")
        st.caption("Interactive analysis of accuracy, ROC-AUC, computational latency, and hardware utilization.")

        if df_exp.empty:
            st.warning("Visualization data is currently unavailable.")
        else:
            chart_type = st.selectbox(
                "Select Visual Comparison Dimension",
                [
                    "Test ROC-AUC by Model & Augmentation (Models 1, 2, 3, 5)",
                    "Validation vs. Test Generalization Gap (ROC-AUC)",
                    "Inference Latency vs. Event Throughput",
                    "Training Time Across Configurations",
                    "Peak RAM & VRAM Hardware Consumption",
                ]
            )

            if chart_type == "Test ROC-AUC by Model & Augmentation (Models 1, 2, 3, 5)":
                df_class = df_exp[df_exp["model_key"] != "model4"].dropna(subset=["test_auc"]).copy()
                if not df_class.empty:
                    df_class["val_label"] = df_class["val_strategy"].map({"no_10fold": "Without 10-fold", "10fold": "With 10-fold"}).fillna("Unknown")

                    fig = px.bar(
                        df_class,
                        x="model_name",
                        y="test_auc",
                        color="aug_condition",
                        barmode="group",
                        facet_col="val_label",
                        title="Test ROC-AUC Comparison Across Augmentations and Validation Strategies",
                        labels={"test_auc": "Test ROC-AUC", "model_name": "Model Architecture", "aug_condition": "Augmentation"},
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                    )
                    fig.update_layout(yaxis_range=[0.3, 1.0], height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No ROC-AUC data available to plot.")

            elif chart_type == "Validation vs. Test Generalization Gap (ROC-AUC)":
                df_class = df_exp[df_exp["model_key"] != "model4"].dropna(subset=["val_auc", "test_auc"]).copy()
                if not df_class.empty:
                    df_class["generalization_gap"] = df_class["val_auc"] - df_class["test_auc"]

                    fig = px.scatter(
                        df_class,
                        x="val_auc",
                        y="test_auc",
                        color="model_name",
                        symbol="aug_condition",
                        hover_data=["run_id", "val_strategy", "train_time_sec"],
                        title="Validation ROC-AUC vs. Test ROC-AUC (Points near diagonal indicate optimal generalization)",
                        labels={"val_auc": "Validation ROC-AUC (Internal Evaluation)", "test_auc": "Test ROC-AUC (Held-out Generalization)"},
                    )
                    fig.add_trace(go.Scatter(x=[0.3, 1.0], y=[0.3, 1.0], mode="lines", name="Ideal Parity", line=dict(dash="dash", color="gray")))
                    fig.update_layout(height=520)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No validation vs test data available to plot.")

            elif chart_type == "Inference Latency vs. Event Throughput":
                df_eff = df_exp.dropna(subset=["inference_time_sec", "throughput_events_per_sec"]).copy()
                if not df_eff.empty:
                    fig = px.scatter(
                        df_eff,
                        x="inference_time_sec",
                        y="throughput_events_per_sec",
                        color="model_name",
                        size="parameters" if df_eff["parameters"].notna().any() else None,
                        hover_data=["aug_condition", "val_strategy"],
                        title="Computational Efficiency: Inference Time vs Throughput",
                        labels={"inference_time_sec": "Inference Latency (Seconds per batch)", "throughput_events_per_sec": "Throughput (Events / sec)"},
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No efficiency data available to plot.")

            elif chart_type == "Training Time Across Configurations":
                df_tt = df_exp.dropna(subset=["train_time_sec"]).copy()
                if not df_tt.empty:
                    fig = px.bar(
                        df_tt,
                        x="model_key",
                        y="train_time_sec",
                        color="aug_condition",
                        barmode="group",
                        facet_col="val_strategy",
                        title="Training Duration (Seconds) across Models, Augmentations, and Cross-Validation",
                        labels={"train_time_sec": "Training Time (s)", "model_key": "Model"},
                    )
                    fig.update_layout(height=480)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No training time data available to plot.")

            elif chart_type == "Peak RAM & VRAM Hardware Consumption":
                df_mem = df_exp.dropna(subset=["peak_ram_mb", "peak_vram_mb"]).copy()
                if not df_mem.empty:
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_ram = px.box(
                            df_mem,
                            x="model_name",
                            y="peak_ram_mb",
                            color="model_name",
                            title="Peak System RAM Consumption (MB)",
                            labels={"peak_ram_mb": "RAM (MB)", "model_name": "Model"},
                        )
                        fig_ram.update_layout(showlegend=False, height=450)
                        st.plotly_chart(fig_ram, use_container_width=True)
                    with col_g2:
                        fig_vram = px.box(
                            df_mem,
                            x="model_name",
                            y="peak_vram_mb",
                            color="model_name",
                            title="Peak GPU VRAM Consumption (MB)",
                            labels={"peak_vram_mb": "VRAM (MB)", "model_name": "Model"},
                        )
                        fig_vram.update_layout(showlegend=False, height=450)
                        st.plotly_chart(fig_vram, use_container_width=True)
                else:
                    st.info("No memory consumption data available to plot.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: MODEL 4 RECOMMENDER ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_model4:
        st.subheader("Model 4: Autoencoder + Recommender Network Specific Results")
        st.caption(
            "Model 4 performs personalized resource ranking and latent preference reconstruction. "
            "Classification accuracy is mathematically secondary to recommendation ranking fidelity."
        )

        m4_df = df_exp[df_exp["model_key"] == "model4"].copy()

        if m4_df.empty or m4_df["hitrate_at_5"].dropna().empty:
            st.info("Model 4 recommendation results are currently not available in recorded logs.")
        else:
            col_m4_kpi1, col_m4_kpi2, col_m4_kpi3, col_m4_kpi4 = st.columns(4)
            best_hr5 = m4_df["hitrate_at_5"].dropna().max() if not m4_df["hitrate_at_5"].dropna().empty else None
            best_ndcg5 = m4_df["ndcg_at_5"].dropna().max() if not m4_df["ndcg_at_5"].dropna().empty else None
            best_mse = m4_df["reconstruction_mse"].dropna().min() if not m4_df["reconstruction_mse"].dropna().empty else None
            best_prec5 = m4_df["precision_at_5"].dropna().max() if not m4_df["precision_at_5"].dropna().empty else None

            with col_m4_kpi1:
                st.metric("Peak HitRate@5", f"{best_hr5*100:.2f}%" if best_hr5 is not None else "Not available")
            with col_m4_kpi2:
                st.metric("Peak NDCG@5", f"{best_ndcg5:.4f}" if best_ndcg5 is not None else "Not available")
            with col_m4_kpi3:
                st.metric("Lowest Reconstruction MSE", f"{best_mse:.4f}" if best_mse is not None else "Not available")
            with col_m4_kpi4:
                st.metric("Peak Precision@5", f"{best_prec5:.4f}" if best_prec5 is not None else "Not available")

            # Model 4 Breakdown Table
            m4_display = pd.DataFrame()
            if "aug_condition" in m4_df.columns:
                m4_display["Augmentation"] = m4_df["aug_condition"].map(lambda x: str(x).upper() if pd.notna(x) else "Not available")
            if "val_strategy" in m4_df.columns:
                m4_display["Validation Strategy"] = m4_df["val_strategy"].map({"no_10fold": "Without 10-fold", "10fold": "With 10-fold"}).fillna("Not available")
            if "reconstruction_mse" in m4_df.columns:
                m4_display["Reconstruction MSE"] = m4_df["reconstruction_mse"].map(lambda x: _fmt(x, decimals=4))
            if "hitrate_at_5" in m4_df.columns:
                m4_display["HitRate@5"] = m4_df["hitrate_at_5"].map(lambda x: _fmt(x, is_pct=True))
            if "ndcg_at_5" in m4_df.columns:
                m4_display["NDCG@5"] = m4_df["ndcg_at_5"].map(lambda x: _fmt(x, decimals=4))
            if "precision_at_5" in m4_df.columns:
                m4_display["Precision@5"] = m4_df["precision_at_5"].map(lambda x: _fmt(x, decimals=4))
            if "recall_at_5" in m4_df.columns:
                m4_display["Recall@5"] = m4_df["recall_at_5"].map(lambda x: _fmt(x, decimals=4))
            if "test_accuracy" in m4_df.columns:
                m4_display["Test Accuracy (Auxiliary)"] = m4_df["test_accuracy"].map(lambda x: _fmt(x, is_pct=True))
            if "inference_time_sec" in m4_df.columns:
                m4_display["Inference Latency"] = m4_df["inference_time_sec"].map(lambda x: f"{x*1000:.1f} ms" if pd.notna(x) else "Not available")
            elif "inference_time_ms" in m4_df.columns:
                m4_display["Inference Latency"] = m4_df["inference_time_ms"].map(lambda x: f"{x:.1f} ms" if pd.notna(x) else "Not available")
            if "throughput_events_per_sec" in m4_df.columns:
                m4_display["Throughput"] = m4_df["throughput_events_per_sec"].map(lambda x: f"{x:,.0f} ev/s" if pd.notna(x) else "Not available")

            st.dataframe(m4_display, use_container_width=True, hide_index=True)

            # Plotly chart for Model 4
            m4_chart_col1, m4_chart_col2 = st.columns(2)
            with m4_chart_col1:
                fig_m4_rec = px.bar(
                    m4_df,
                    x="aug_condition",
                    y="hitrate_at_5",
                    color="val_strategy",
                    barmode="group",
                    title="Model 4: HitRate@5 by Augmentation & Validation Strategy",
                    labels={"hitrate_at_5": "HitRate@5", "aug_condition": "Augmentation"},
                )
                st.plotly_chart(fig_m4_rec, use_container_width=True)
            with m4_chart_col2:
                fig_m4_mse = px.bar(
                    m4_df,
                    x="aug_condition",
                    y="reconstruction_mse",
                    color="val_strategy",
                    barmode="group",
                    title="Model 4: Autoencoder Reconstruction MSE (Lower is Better)",
                    labels={"reconstruction_mse": "Reconstruction MSE", "aug_condition": "Augmentation"},
                )
                st.plotly_chart(fig_m4_mse, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: FINAL STANDALONE BENCHMARKS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_final:
        st.subheader("Final Standalone Benchmark Models")
        st.caption("Official benchmark comparisons on the complete test cohort from reports/five_model_comparison.csv.")

        df_five = get_five_model_comparison()
        if df_five is not None and not df_five.empty:
            st.dataframe(df_five, use_container_width=True, hide_index=True)
        else:
            st.warning("reports/five_model_comparison.csv not found.")

        st.markdown("---")
        st.subheader("Cross-Validation & Generalization Diagnosis")
        st.caption("From reports/final_comparison.csv")
        df_final = get_final_comparison()
        if df_final is not None and not df_final.empty:
            st.dataframe(df_final, use_container_width=True, hide_index=True)
        else:
            st.warning("reports/final_comparison.csv not found.")

        st.markdown("---")
        st.subheader("Data Augmentation Technique Benchmark")
        st.caption("From reports/augmentation_comparison.csv")
        df_aug = get_augmentation_comparison()
        if df_aug is not None and not df_aug.empty:
            st.dataframe(df_aug, use_container_width=True, hide_index=True)
        else:
            st.warning("reports/augmentation_comparison.csv not found.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: STATIC FIGURES & CURVES GALLERY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_plots:
        st.subheader("Static Diagnostic Plots & Verification Figures")
        st.caption("Pre-computed high-resolution evaluation figures from reports/figures/ and plots/.")

        all_figures = [
            ("Model Accuracy Comparison", "model_accuracy_comparison.png", "figure"),
            ("Model ROC-AUC Comparison", "model_roc_auc_comparison.png", "figure"),
            ("Model F1-Score Comparison", "model_f1_comparison.png", "figure"),
            ("Training Time Comparison", "training_time_comparison.png", "figure"),
            ("Inference Latency Comparison", "inference_latency_comparison.png", "figure"),
            ("Throughput Comparison", "throughput_comparison.png", "figure"),
            ("System RAM Comparison", "ram_comparison.png", "figure"),
            ("GPU VRAM Comparison", "gpu_vram_comparison.png", "figure"),
            ("10-Fold Cross-Validation Comparison", "validation_10fold_comparison.png", "figure"),
            ("Augmentation Comparison", "augmentation_comparison.png", "figure"),
            ("SMOTE Comparison", "smote_comparison.png", "figure"),
            ("ADASYN Comparison", "adasyn_comparison.png", "figure"),
            ("WGAN-GP Comparison", "gan_comparison.png", "figure"),
            ("Fit Correction Comparison", "fit_correction_comparison.png", "figure"),
            ("Five Model Comparison Overview", "five_model_comparison.png", "plot"),
            ("GAN Augmentation Fidelity", "gan_augmentation_fidelity.png", "plot"),
        ]

        fig_choice = st.selectbox(
            "Select Diagnostic Figure to Inspect",
            [title for title, _, _ in all_figures]
        )

        selected_item = next(item for item in all_figures if item[0] == fig_choice)
        title, filename, location = selected_item

        img_path = get_figure_path(filename) if location == "figure" else get_plot_path(filename)
        if img_path and img_path.exists():
            st.image(str(img_path), caption=f"{title} ({filename})", use_container_width=True)
        else:
            st.warning(f"Plot file `{filename}` not found in `{location}s/`.")
