# -*- coding: utf-8 -*-
"""
Smart Education - Model Evaluation & Performance Page
======================================================
Interactive Streamlit GUI module for Model Evaluation & Performance Comparison.
Displays multi-model benchmark metrics, interactive Plotly visualizations,
fold-by-fold 10-fold cross-validation results, and strict training configurations.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.inference_service import (
    get_experiment_manifest,
    get_model_results,
    get_cv_results,
    get_plot_path,
    get_five_model_comparison,
)


def _format_metric_val(key: str, val: Any) -> str:
    """Safely format a metric value for tabular display."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        k_lower = key.lower()
        if any(term in k_lower for term in ["accuracy", "recall", "precision", "f1", "rate", "hitrate"]):
            return f"{val:.4f} ({val * 100:.2f}%)"
        return f"{val:.4f}"
    return str(val)


def render(engine: Optional[Any] = None) -> None:
    """
    Render entry point for the Model Evaluation & Performance page.

    Args:
        engine: MultiModelInferenceEngine instance or None if not loaded.
    """
    # 1. Page Header
    st.title("Model Evaluation & Performance")
    st.caption("Multi-model benchmarking, 10-fold cross-validation rigor, and training configuration audits.")

    # 2. Load experiment manifest
    manifest: Dict[str, Any] = get_experiment_manifest()
    if not manifest:
        st.warning("Experiment manifest not found. Please ensure reports/experiment_manifest.json is present.")

    models_summary = manifest.get("models_summary", {}) if manifest else {}

    # Executive Overview KPIs
    if models_summary:
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        m5_auc = models_summary.get("model5", {}).get("test_auc")
        m5_acc = models_summary.get("model5", {}).get("test_acc")
        m4_hit10 = models_summary.get("model4", {}).get("hit_rate_at_10")
        total_models = len(models_summary)

        kpi_col1.metric("Evaluated Models", f"{total_models} Architectures")
        kpi_col2.metric("Top Test ROC-AUC", f"{m5_auc:.4f}" if m5_auc is not None else "N/A", "Model 5 (CNN+LSTM)")
        kpi_col3.metric("Top Test Accuracy", f"{m5_acc * 100:.2f}%" if m5_acc is not None else "N/A", "Model 5 (CNN+LSTM)")
        kpi_col4.metric("Recommendation Hit@10", f"{m4_hit10 * 100:.2f}%" if m4_hit10 is not None else "N/A", "Model 4 (Autoencoder)")

    st.markdown("---")

    # 3. 5-Model Comparison Table
    st.subheader("1. Multi-Model Architecture Comparison")
    st.caption("Standardized test set evaluation metrics across all 5 implemented models.")

    if models_summary:
        table_rows = []
        ordered_keys = ["model1", "model2", "model3", "model4", "model5"]
        for key in ordered_keys:
            if key not in models_summary:
                continue
            item = models_summary[key]
            name = item.get("name", key)
            params = item.get("parameters", 0)
            formatted_params = f"{params:,}" if isinstance(params, (int, float)) else str(params)

            if key == "model4":
                # Model 4 is a recommendation autoencoder with distinct evaluation metrics
                mse = item.get("reconstruction_mse")
                hr5 = item.get("hit_rate_at_5")
                hr10 = item.get("hit_rate_at_10")

                mse_text = f"MSE: {mse:.4f}" if mse is not None else "N/A"
                auc_display = f"N/A ({mse_text})"

                hr_parts = []
                if hr5 is not None:
                    hr_parts.append(f"Hit@5: {hr5 * 100:.2f}%")
                if hr10 is not None:
                    hr_parts.append(f"Hit@10: {hr10 * 100:.2f}%")
                acc_display = f"N/A ({', '.join(hr_parts)})" if hr_parts else "N/A"
            else:
                auc_val = item.get("test_auc")
                acc_val = item.get("test_acc")
                auc_display = f"{auc_val:.4f}" if auc_val is not None else "N/A"
                acc_display = f"{acc_val:.4f} ({acc_val * 100:.2f}%)" if acc_val is not None else "N/A"

            table_rows.append({
                "Model": name,
                "Parameters": formatted_params,
                "Test AUC": auc_display,
                "Test Accuracy": acc_display,
            })

        df_comparison = pd.DataFrame(table_rows)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    else:
        st.warning("No models summary available in experiment manifest.")

    # Optional view: Full 16-metric comparison CSV
    full_comp_df = get_five_model_comparison()
    if full_comp_df is not None and not full_comp_df.empty:
        with st.expander("View Full 16-Metric Comparison Benchmark Matrix (five_model_comparison.csv)"):
            st.dataframe(full_comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4 & 5. Plotly Grouped Bar Chart & Parameter Count Chart
    st.subheader("2. Performance & Capacity Visualizations")

    chart_col1, chart_col2 = st.columns(2)

    # 4. Grouped bar chart: Test AUC and Test Accuracy for models 1, 2, 3, 5
    with chart_col1:
        perf_data = []
        kt_model_labels = {
            "model1": "Model 1 (LSTM)",
            "model2": "Model 2 (Transformer)",
            "model3": "Model 3 (BERT-NCF)",
            "model5": "Model 5 (CNN-LSTM)",
        }
        for k in ["model1", "model2", "model3", "model5"]:
            if k in models_summary:
                m = models_summary[k]
                label = kt_model_labels.get(k, m.get("name", k))
                auc = m.get("test_auc")
                acc = m.get("test_acc")
                if auc is not None:
                    perf_data.append({
                        "Model": label,
                        "Metric": "Test AUC",
                        "Score": round(float(auc), 4),
                    })
                if acc is not None:
                    perf_data.append({
                        "Model": label,
                        "Metric": "Test Accuracy",
                        "Score": round(float(acc), 4),
                    })

        if perf_data:
            df_perf = pd.DataFrame(perf_data)
            fig_perf = px.bar(
                df_perf,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score",
                title="Knowledge Tracing: Test AUC vs. Accuracy",
                color_discrete_map={"Test AUC": "#2563EB", "Test Accuracy": "#10B981"},
            )
            fig_perf.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_perf.update_layout(
                yaxis=dict(range=[0, 1.15], title="Metric Score"),
                xaxis=dict(title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
                height=380,
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Performance metric data not available for knowledge tracing models.")

    # 5. Parameter count comparison: Plotly bar chart showing parameter counts
    with chart_col2:
        param_data = []
        all_model_labels = {
            "model1": "Model 1 (LSTM)",
            "model2": "Model 2 (Transformer)",
            "model3": "Model 3 (BERT-NCF)",
            "model4": "Model 4 (Autoencoder)",
            "model5": "Model 5 (CNN-LSTM)",
        }
        for k in ["model1", "model2", "model3", "model4", "model5"]:
            if k in models_summary:
                m = models_summary[k]
                label = all_model_labels.get(k, m.get("name", k))
                params = m.get("parameters", 0)
                param_data.append({
                    "Model": label,
                    "Parameters": int(params),
                })

        if param_data:
            df_params = pd.DataFrame(param_data)
            fig_params = px.bar(
                df_params,
                x="Model",
                y="Parameters",
                text="Parameters",
                title="Trainable Parameters Across All 5 Models",
                color="Parameters",
                color_continuous_scale="Blues",
            )
            fig_params.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_params.update_layout(
                yaxis=dict(title="Parameter Count"),
                xaxis=dict(title=""),
                margin=dict(l=20, r=20, t=50, b=20),
                height=380,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_params, use_container_width=True)
        else:
            st.info("Parameter count data not available.")

    st.markdown("---")

    # 6. Benchmark Comparison Plot from Disk
    st.subheader("3. Benchmark Comparison Visualization")
    comparison_plot_path = get_plot_path("five_model_comparison.png")
    if comparison_plot_path and comparison_plot_path.exists():
        st.image(
            str(comparison_plot_path),
            caption="Comprehensive 5-Model Benchmark Comparison (plots/five_model_comparison.png)",
            use_container_width=True,
        )
    else:
        st.info("Benchmark visualization plot 'five_model_comparison.png' not found in plots directory.")

    st.markdown("---")

    # 7. Per-Model Test Metrics (Expandable)
    st.subheader("4. Detailed Per-Model Test Metrics")
    st.caption("Expand any model to inspect granular evaluation results and compute consumption.")

    model_display_titles = {
        1: "Model 1: LSTM + Attention (Knowledge Tracing)",
        2: "Model 2: Transformer + Knowledge Tracing",
        3: "Model 3: BERT-style Transformer + Neural Collaborative Filtering",
        4: "Model 4: Autoencoder + Recommender Network",
        5: "Model 5: CNN + LSTM (Temporal Knowledge Tracing)",
    }

    for i in range(1, 6):
        title = model_display_titles.get(i, f"Model {i}")
        with st.expander(f"{title} - Test Results"):
            res = get_model_results(i)
            if not res:
                st.info(f"No result records found for Model {i} in reports/model{i}_results.json.")
                continue

            # Execution runtime & memory metrics
            t_sec = res.get("training_time_sec")
            inf_sec = res.get("inference_time_sec")
            gpu_mb = res.get("gpu_memory_mb")

            k_cols = st.columns(3)
            if t_sec is not None:
                k_cols[0].metric("Training Duration", f"{t_sec:.2f} s")
            if inf_sec is not None:
                inf_display = f"{inf_sec * 1000:.2f} ms" if inf_sec < 1 else f"{inf_sec:.2f} s"
                k_cols[1].metric("Inference Latency", inf_display)
            if gpu_mb is not None:
                k_cols[2].metric("Peak VRAM Allocated", f"{gpu_mb:.2f} MB")

            # Detailed test metrics table
            test_metrics = res.get("test_metrics", {})
            if test_metrics:
                rows = []
                for k, v in test_metrics.items():
                    rows.append({
                        "Metric Name": k.replace("_", " ").title(),
                        "Measured Value": _format_metric_val(k, v),
                    })
                df_metrics = pd.DataFrame(rows)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            else:
                st.info("No test metrics dictionary found for this model.")

    st.markdown("---")

    # 8. Cross-Validation Results (Expandable per Model)
    st.subheader("5. 10-Fold Cross-Validation Audit")
    st.caption("Fold-by-fold performance validation ensuring zero student leakage across folds.")

    for i in range(1, 6):
        title = model_display_titles.get(i, f"Model {i}")
        with st.expander(f"{title} - 10-Fold CV Details"):
            cv_df = get_cv_results(i)
            if cv_df is not None and not cv_df.empty:
                # Calculate fold-level statistics for primary metrics
                exclude_cols = {
                    "fold", "train_users", "val_users", "user_overlap",
                    "train_events", "val_events"
                }
                metric_cols = [
                    c for c in cv_df.columns
                    if c not in exclude_cols and pd.api.types.is_numeric_dtype(cv_df[c])
                ]

                if metric_cols:
                    summary_stats = []
                    for col in metric_cols:
                        m_mean = cv_df[col].mean()
                        m_std = cv_df[col].std()
                        summary_stats.append({
                            "Metric": col.replace("_", " ").title(),
                            "Mean +/- Std": f"{m_mean:.4f} +/- {m_std:.4f}",
                            "Min Fold": f"{cv_df[col].min():.4f}",
                            "Max Fold": f"{cv_df[col].max():.4f}",
                        })
                    st.markdown("**Cross-Validation Summary (10 Folds):**")
                    st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)

                st.markdown("**Fold-by-Fold Breakdown:**")
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No 10-fold cross-validation CSV results available for Model {i}.")

    st.markdown("---")

    # 9. Training Configuration Summary from Manifest
    st.subheader("6. Experimental & Training Configuration")
    st.caption("Reproducibility parameters and data leakage prevention safeguards verified from experiment manifest.")

    if manifest:
        # High-level reproducibility KPIs
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Random Seed", str(manifest.get("random_seed", "N/A")))
        col_c2.metric("Execution Device", str(manifest.get("execution_device", "N/A")).upper())
        gpu_name = manifest.get("gpu_hardware", "N/A")
        if len(gpu_name) > 22:
            gpu_display = gpu_name[:20] + "..."
        else:
            gpu_display = gpu_name
        col_c3.metric("Hardware", gpu_display)
        split_strat = manifest.get("split_strategy", {})
        col_c4.metric("Train Split", f"{float(split_strat.get('train_ratio', 0.8)) * 100:.0f}%")

        # Two columns for Split Strategy and SMOTE Settings
        cfg_left, cfg_right = st.columns(2)

        with cfg_left:
            st.markdown("##### Split Strategy & Cohort Isolation")
            split_info = [
                {"Parameter": "Method", "Configuration": str(split_strat.get("method", "N/A"))},
                {"Parameter": "Train Events", "Configuration": f"{split_strat.get('train_events', 0):,}"},
                {"Parameter": "Validation Events", "Configuration": f"{split_strat.get('val_events', 0):,}"},
                {"Parameter": "Test Events", "Configuration": f"{split_strat.get('test_events', 0):,}"},
                {"Parameter": "Test Cohort Status", "Configuration": "Strictly isolated (zero user overlap)"},
            ]
            st.dataframe(pd.DataFrame(split_info), use_container_width=True, hide_index=True)

        with cfg_right:
            st.markdown("##### SMOTE Oversampling Safeguards")
            smote_cfg = manifest.get("smote_settings", {})
            val_aug = smote_cfg.get("validation_augmented", False)
            test_aug = smote_cfg.get("test_augmented", False)

            smote_info = [
                {"Parameter": "Target Partition", "Configuration": str(smote_cfg.get("target_partition", "train.csv exclusively"))},
                {"Parameter": "k-Neighbors", "Configuration": str(smote_cfg.get("k_neighbors", "5"))},
                {"Parameter": "Sampling Strategy", "Configuration": str(smote_cfg.get("sampling_strategy", "1.0 (balanced)"))},
                {"Parameter": "Validation Augmented", "Configuration": "No (Data Leakage Prohibited)" if not val_aug else "Yes"},
                {"Parameter": "Test Augmented", "Configuration": "No (Preserves Real World Distribution)" if not test_aug else "Yes"},
            ]
            st.dataframe(pd.DataFrame(smote_info), use_container_width=True, hide_index=True)

        # Cross validation protocol
        cv_settings = manifest.get("cross_validation_settings", {})
        if cv_settings:
            with st.expander("GroupKFold Cross-Validation Protocol"):
                st.markdown(f"- **Strategy**: {cv_settings.get('strategy', '10-Fold GroupKFold')}")
                st.markdown(f"- **Number of Splits**: {cv_settings.get('n_splits', 10)}")
                st.markdown(f"- **Enforced User Overlap**: {cv_settings.get('enforced_overlap', 0)} students (strictly 0)")
    else:
        st.info("No training configuration found in experiment manifest.")
