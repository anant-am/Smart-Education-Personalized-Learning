"""
Smart Education - Data Augmentation & Class Imbalance Analysis Page
====================================================================
Comprehensive empirical evaluation of resampling and generative paradigms:
- Baseline (Natural imbalanced distribution: 33.2% Incorrect vs 66.8% Correct)
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- Tabular WGAN-GP (Conditional Wasserstein GAN with Gradient Penalty)

Academic Guarantees:
- Zero Data Leakage: Augmentation performed exclusively on train.csv.
- Natural Test Evaluation: Val and Test sets remain 100% un-augmented.
"""

from pathlib import Path
from typing import Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from services.inference_service import (
    get_augmentation_comparison,
    get_report_markdown,
    get_plot_path,
)


def render(engine=None):
    """
    Renders the Data Augmentation & Class Imbalance Analysis page.

    Args:
        engine: MultiModelInferenceEngine instance (or None if unavailable).
    """
    # -- 1. Page Header --
    st.title("Data Augmentation")
    st.caption(
        "Benchmarking resampling and generative strategies to resolve severe class imbalance "
        "in student interaction sequences."
    )

    # -- 2. Overview: The Class Imbalance Problem --
    st.header("1. The Class Imbalance Problem in Learning Analytics")

    # Overview KPI Metrics
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(
            label="Class 0 (Incorrect / Struggling)",
            value="141,257",
            delta="33.2% (Minority)",
            delta_color="inverse",
        )
    with col_kpi2:
        st.metric(
            label="Class 1 (Correct / Mastery)",
            value="284,418",
            delta="66.8% (Majority)",
            delta_color="normal",
        )
    with col_kpi3:
        st.metric(
            label="Natural Imbalance Ratio",
            value="0.497 : 1",
            delta="Approx. 1 : 2 ratio",
            delta_color="off",
        )
    with col_kpi4:
        st.metric(
            label="Baseline Minority Recall",
            value="0.0232",
            delta="97.7% Failure Miss Rate",
            delta_color="inverse",
        )

    # Narrative explanation with academic context
    with st.expander("Theoretical Problem Definition & Academic Defense", expanded=True):
        st.markdown(
            """
            ### Why Class Imbalance Breaks Standard Educational AI:
            - **Natural Positive Skew:** Real-world educational logs suffer from acute positive skew 
              (**66.8% correct** vs **33.2% incorrect** in `train.csv`). Students generally answer more items correctly than incorrectly.
            - **The Deceptive Accuracy Trap:** Left uncorrected, standard machine learning and deep learning models 
              exploit this skew by heavily favoring majority-class predictions. The unaugmented baseline model achieves a deceptively 
              respectable **67.08% overall accuracy**, yet its recall on struggling student responses (Class 0) is an abysmal **0.0232 (2.32%)**.
            - **Pedagogical Catastrophe:** Missing **97.68%** of student failure events defeats the purpose of an AI tutor. 
              The system fails to trigger personalized hints, remediation exercises, or teacher alerts when students are actually struggling.
            - **Strict Zero-Leakage Guarantee:** All three synthetic augmentation techniques (SMOTE, ADASYN, WGAN-GP) 
              were fitted and synthesized **exclusively on `train.csv`**. Both the validation fold (`val.csv`) and the 
              250-student evaluation cohort (`test.csv` - 174,608 events) remain **100% untouched** in their authentic 
              natural distribution, guaranteeing uncompromised academic evaluation integrity.
            """
        )

    st.divider()

    # -- 3. 4-Regime Comparison Table & Plotly Analysis --
    st.header("2. 4-Regime Benchmark Comparison")
    st.markdown(
        "Empirical performance evaluated across all four paradigms on the strictly isolated "
        "`test.csv` cohort (249 valid students, 174,608 chronological events)."
    )

    df_aug = get_augmentation_comparison()

    if df_aug is None or df_aug.empty:
        st.warning("Augmentation comparison data file ('reports/augmentation_comparison.csv') not found.")
    else:
        # Create tabs for Table view and Interactive Metric Visualizer
        tab_table, tab_chart = st.tabs(["Benchmark Table (Styled)", "Interactive Plotly Visualizer"])

        with tab_table:
            # Format numbers for clean presentation
            fmt_dict = {
                "Train Samples": "{:,}",
                "Class 0 Count": "{:,}",
                "Class 1 Count": "{:,}",
                "Imbalance Ratio (0:1)": "{:.3f}",
                "Train Time (s)": "{:.2f} s",
                "Test Accuracy": "{:.4f}",
                "Test ROC-AUC": "{:.4f}",
                "Test PR-AUC": "{:.4f}",
                "Test LogLoss": "{:.4f}",
                "Test Macro F1": "{:.4f}",
                "Test Minority Recall (Class 0)": "{:.4f}",
                "Test Minority Precision (Class 0)": "{:.4f}",
                "Test Minority F1 (Class 0)": "{:.4f}",
                "Val Accuracy": "{:.4f}",
                "Val ROC-AUC": "{:.4f}",
                "Val Minority Recall": "{:.4f}",
                "Val Minority F1": "{:.4f}",
            }

            active_fmt = {k: v for k, v in fmt_dict.items() if k in df_aug.columns}
            styled_df = df_aug.style.format(active_fmt)

            # Highlight best minority recall and macro F1
            if "Test Minority Recall (Class 0)" in df_aug.columns:
                styled_df = styled_df.highlight_max(
                    subset=["Test Minority Recall (Class 0)"],
                    color="#d1fae5"
                )
            if "Test Macro F1" in df_aug.columns:
                styled_df = styled_df.highlight_max(
                    subset=["Test Macro F1"],
                    color="#e0e7ff"
                )

            st.dataframe(styled_df, use_container_width=True)
            st.caption("Green highlight: Highest Minority Recall (Class 0) | Lavender highlight: Highest Macro F1.")

        with tab_chart:
            # Interactive Bar Chart: Test Minority Recall across regimes
            recall_col = "Test Minority Recall (Class 0)"
            if recall_col in df_aug.columns:
                fig_recall = go.Figure()

                colors = ["#ef4444", "#3b82f6", "#10b981", "#8b5cf6"]
                regimes = df_aug["Technique"].tolist()
                recalls = df_aug[recall_col].tolist()

                fig_recall.add_trace(go.Bar(
                    x=regimes,
                    y=recalls,
                    text=[f"{val:.4f} ({val*100:.1f}%)" for val in recalls],
                    textposition="auto",
                    textfont=dict(size=13, color="white"),
                    marker=dict(
                        color=colors[:len(regimes)],
                        line=dict(color="#1f2937", width=1.5)
                    ),
                    hovertemplate="<b>%{x}</b><br>Minority Recall: %{y:.4f}<extra></extra>"
                ))

                # Add dashed reference line at Baseline
                baseline_val = recalls[0] if len(recalls) > 0 else 0.0232
                fig_recall.add_hline(
                    y=baseline_val,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text=f"Baseline Failure: {baseline_val:.4f} (2.3%)",
                    annotation_position="top left",
                )

                fig_recall.update_layout(
                    title="<b>Test Minority Recall (Class 0: Incorrect) Across 4 Augmentation Regimes</b>",
                    xaxis_title="<b>Augmentation Paradigm</b>",
                    yaxis_title="<b>Minority Recall (Catching Student Struggle)</b>",
                    yaxis=dict(range=[0, 0.65], tickformat=".2f"),
                    template="plotly_white",
                    height=450,
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig_recall, use_container_width=True)

            # Secondary Multi-Metric Comparison Chart
            st.markdown("#### Multi-Metric Comparison across Augmentation Regimes")
            metric_options = [
                "Test Minority Recall (Class 0)",
                "Test Minority F1 (Class 0)",
                "Test Macro F1",
                "Test ROC-AUC",
                "Test PR-AUC",
                "Test Accuracy",
            ]
            available_metrics = [m for m in metric_options if m in df_aug.columns]

            selected_metrics = st.multiselect(
                "Select metrics to compare:",
                options=available_metrics,
                default=[
                    m for m in [
                        "Test Minority Recall (Class 0)",
                        "Test Minority F1 (Class 0)",
                        "Test Macro F1",
                    ] if m in available_metrics
                ]
            )

            if selected_metrics:
                fig_multi = go.Figure()
                palette = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899", "#6b7280"]

                for idx, metric in enumerate(selected_metrics):
                    fig_multi.add_trace(go.Bar(
                        name=metric.replace(" (Class 0)", ""),
                        x=df_aug["Technique"],
                        y=df_aug[metric],
                        text=[f"{v:.3f}" for v in df_aug[metric]],
                        textposition="inside",
                        marker_color=palette[idx % len(palette)],
                        hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
                    ))

                fig_multi.update_layout(
                    barmode="group",
                    title="<b>Cross-Regime Performance Comparison by Selected Metrics</b>",
                    xaxis_title="<b>Technique</b>",
                    yaxis_title="<b>Score</b>",
                    yaxis=dict(range=[0, 1.0], tickformat=".2f"),
                    template="plotly_white",
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig_multi, use_container_width=True)

    st.divider()

    # -- 4. Key Findings Summary --
    st.header("3. Key Findings Summary")

    card1, card2, card3, card4 = st.columns(4)
    with card1:
        st.markdown(
            """
            <div style="background-color: #ecfdf5; border: 1px solid #10b981; border-radius: 8px; padding: 14px;">
                <h4 style="color: #065f46; margin: 0 0 6px 0;">[Rank 1] ADASYN (Best)</h4>
                <p style="font-size: 1.4rem; font-weight: 700; color: #047857; margin: 0;">0.5582</p>
                <p style="color: #065f46; font-size: 0.85rem; margin: 4px 0 0 0;">
                    <b>Best Minority Recall:</b> 24.1x boost over baseline. Adaptively generates samples at difficult decision boundaries.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with card2:
        st.markdown(
            """
            <div style="background-color: #eff6ff; border: 1px solid #3b82f6; border-radius: 8px; padding: 14px;">
                <h4 style="color: #1e40af; margin: 0 0 6px 0;">[Rank 2] SMOTE (Close 2nd)</h4>
                <p style="font-size: 1.4rem; font-weight: 700; color: #1d4ed8; margin: 0;">0.5536</p>
                <p style="color: #1e40af; font-size: 0.85rem; margin: 4px 0 0 0;">
                    <b>Minority Recall:</b> 23.9x boost over baseline. Uniform linear k-NN interpolation yields balanced, consistent gains.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with card3:
        st.markdown(
            """
            <div style="background-color: #f5f3ff; border: 1px solid #8b5cf6; border-radius: 8px; padding: 14px;">
                <h4 style="color: #5b21b6; margin: 0 0 6px 0;">[Rank 3] Tabular WGAN-GP</h4>
                <p style="font-size: 1.4rem; font-weight: 700; color: #6d28d9; margin: 0;">0.4625</p>
                <p style="color: #5b21b6; font-size: 0.85rem; margin: 4px 0 0 0;">
                    <b>Minority Recall:</b> 19.9x boost over baseline. Generates realistic, non-linear student interaction vectors via neural generator.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with card4:
        st.markdown(
            """
            <div style="background-color: #fef2f2; border: 1px solid #ef4444; border-radius: 8px; padding: 14px;">
                <h4 style="color: #991b1b; margin: 0 0 6px 0;">[Baseline] Imbalanced</h4>
                <p style="font-size: 1.4rem; font-weight: 700; color: #b91c1c; margin: 0;">0.0232</p>
                <p style="color: #991b1b; font-size: 0.85rem; margin: 4px 0 0 0;">
                    <b>Critical Blind Spot:</b> Misses 97.7% of failure states despite deceptively high 67.1% accuracy.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Analytical Breakdown: The Accuracy vs Recall Tradeoff", expanded=False):
        st.markdown(
            """
            ### Understanding the Pedagogical Value of Resampling:
            1. **The Tradeoff:**
               - Baseline Accuracy: **67.08%** | Minority Recall: **0.0232**
               - ADASYN Accuracy: **58.30%** | Minority Recall: **0.5582**
               - By sacrificing ~8.8% overall accuracy, ADASYN achieves a **24.1-fold surge** in identifying student struggle states.
            2. **Why This Matters for Smart Education:**
               - In standard classification, overall accuracy is often treated as the primary KPI.
               - In educational recommender systems, **false negatives** (predicting a struggling student is doing fine) 
                 result in abandoned students who fall further behind.
               - **False positives** (predicting a student might struggle when they might have passed) merely lead to offering 
                 an extra explanatory hint or easier practice item, which is pedagogically harmless.
            3. **Theoretical Distinction Between Methods:**
               - **SMOTE** connects random minority k-nearest neighbors uniformly, sometimes bridging into majority-dense pockets.
               - **ADASYN** calculates the proportion of majority neighbors for each minority point and focuses synthesis on 
                 ambiguous decision boundaries where students transition from understanding to confusion.
               - **WGAN-GP** learns the global continuous joint feature distribution with gradient penalty constraints, ensuring 
                 synthesized interactions adhere to realistic student feature covariances.
            """
        )

    st.divider()

    # -- 5. Individual Augmentation Reports (st.tabs) --
    st.header("4. Individual Augmentation Technical Reports")
    st.markdown("Detailed methodological guarantees, hyperparameters, and quantitative class rebalancing logs.")

    tab_smote, tab_adasyn, tab_gan = st.tabs([
        "SMOTE Technical Report",
        "ADASYN Technical Report",
        "Tabular WGAN-GP Technical Report",
    ])

    with tab_smote:
        smote_md = get_report_markdown("smote_report.md")
        if smote_md:
            st.markdown(smote_md)
        else:
            st.warning("SMOTE report file ('reports/smote_report.md') not found.")

    with tab_adasyn:
        adasyn_md = get_report_markdown("adasyn_report.md")
        if adasyn_md:
            st.markdown(adasyn_md)
        else:
            st.warning("ADASYN report file ('reports/adasyn_report.md') not found.")

    with tab_gan:
        gan_md = get_report_markdown("gan_augmentation_report.md")
        if gan_md:
            st.markdown(gan_md)
        else:
            st.warning("WGAN-GP report file ('reports/gan_augmentation_report.md') not found.")

    st.divider()

    # -- 6. Visualizations & Statistical Fidelity --
    st.header("5. Empirical Visualizations & Fidelity Assessment")
    st.markdown(
        "Published high-resolution evaluation figures comparing cross-regime performance "
        "and generative fidelity metrics."
    )

    plot_aug_path = get_plot_path("augmentation_comparison.png")
    plot_gan_path = get_plot_path("gan_augmentation_fidelity.png")

    tab_vis1, tab_vis2 = st.tabs([
        "4-Regime Benchmark Plot",
        "WGAN-GP Feature Fidelity & Statistical Validation",
    ])

    with tab_vis1:
        if plot_aug_path and plot_aug_path.exists():
            st.image(
                str(plot_aug_path),
                caption="Figure 1: Comparative Evaluation of Resampling Paradigms across Accuracy, ROC-AUC, PR-AUC, and Minority Recall.",
                use_container_width=True,
            )
        else:
            st.info("Plot file 'plots/augmentation_comparison.png' not found.")

    with tab_vis2:
        if plot_gan_path and plot_gan_path.exists():
            st.image(
                str(plot_gan_path),
                caption="Figure 2: Conditional Tabular WGAN-GP Feature Fidelity: Wasserstein-1 Distances and Empirical Cumulative Distributions.",
                use_container_width=True,
            )
            st.markdown(
                """
                **Generative Statistical Quality Interpretation:**
                - **Mean Wasserstein Distance:** `55.65` across all 10 tabular feature spaces.
                - **Correlation Frobenius Norm:** `0.6426`, confirming that multi-feature correlation structures 
                  (such as response time vs question difficulty) are preserved without mode collapse.
                - **Continuous Joint Manifold:** Unlike linear interpolation, WGAN-GP preserves non-linear topological 
                  dependencies between historical accuracy and sequential latency.
                """
            )
        else:
            st.info("Plot file 'plots/gan_augmentation_fidelity.png' not found.")
