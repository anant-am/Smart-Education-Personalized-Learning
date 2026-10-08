"""
Smart Education - Dataset Explorer & Data Quality
==================================================
Interactive GUI module for exploring dataset splits, cohort statistics,
data provenance, leakage audits, and feature definitions.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.inference_service import (
    get_experiment_manifest,
    get_dataset_stats,
    get_dataset_user_counts,
    get_dataset_sample,
    get_report_markdown,
)


def render(engine=None):
    """
    Entry point for the Dataset Explorer & Data Quality page.
    
    Args:
        engine: MultiModelInferenceEngine instance or None.
    """
    # -- 1. Page Header --
    st.title("Dataset Explorer")
    st.caption("Provenance audit, partition metrics, leakage verifications, and feature inspection across EdNet-KT3 cohorts.")

    # -- 2. Cohort Overview --
    st.subheader("1. Cohort Overview")
    manifest = get_experiment_manifest()
    
    if not manifest:
        st.warning("Experiment manifest (experiment_manifest.json) not found or empty.")
    else:
        cohort_stats = manifest.get("cohort_statistics", {})
        dev_users = cohort_stats.get("development_users", 0)
        dev_events = cohort_stats.get("development_events", 0)
        test_users = cohort_stats.get("final_test_users", 0)
        test_events = cohort_stats.get("final_test_events", 0)
        overlap = cohort_stats.get("cross_cohort_user_overlap", 0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            label="Dev Cohort Users",
            value=f"{dev_users:,}" if isinstance(dev_users, (int, float)) else str(dev_users),
            help="Unique learners in the KT-3 (1000) development cohort"
        )
        c2.metric(
            label="Dev Cohort Events",
            value=f"{dev_events:,}" if isinstance(dev_events, (int, float)) else str(dev_events),
            help="Total interaction events in the development cohort"
        )
        c3.metric(
            label="Test Cohort Users",
            value=f"{test_users:,}" if isinstance(test_users, (int, float)) else str(test_users),
            help="Unique unseen learners in the KT-3 (250) test cohort"
        )
        c4.metric(
            label="Test Cohort Events",
            value=f"{test_events:,}" if isinstance(test_events, (int, float)) else str(test_events),
            help="Total interaction events in the final unseen test cohort"
        )
        c5.metric(
            label="Cross-Cohort Overlap",
            value=str(overlap),
            delta="Zero Leakage (Passed)" if overlap == 0 else "Leakage Detected!",
            delta_color="normal" if overlap == 0 else "inverse",
            help="Enforced zero student overlap between development and test cohorts"
        )

        # Visual Cohort Summary
        col_summary1, col_summary2 = st.columns([3, 2])
        with col_summary1:
            st.markdown(
                "**Cohort Isolation Architecture:**\n"
                "- **Development Cohort (`KT-3 1000`):** Contains 998 active students used exclusively for "
                "training, cross-validation, and validation splitting.\n"
                "- **Final Test Cohort (`KT-3 250 TEST`):** Contains 249 completely isolated, unseen students "
                "held out until final model benchmarking.\n"
                "- **Quarantine Rule:** All encoders, vocabularies (11,335 questions, 189 concept tags), "
                "and response time scalers were strictly fitted on the Development Cohort alone."
            )
        with col_summary2:
            cohort_df = pd.DataFrame([
                {"Cohort": "Development", "Students": dev_users, "Interactions": dev_events},
                {"Cohort": "Final Test", "Students": test_users, "Interactions": test_events},
            ])
            fig_cohort = px.bar(
                cohort_df,
                x="Cohort",
                y="Interactions",
                text="Interactions",
                color="Cohort",
                color_discrete_sequence=["#2563EB", "#10B981"],
                title="Interactions by Source Cohort"
            )
            fig_cohort.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_cohort.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_cohort, use_container_width=True)

    st.divider()

    # -- 3. Split Statistics --
    st.subheader("2. Dataset Split Statistics")
    stats = get_dataset_stats()
    user_counts = get_dataset_user_counts()

    if not stats:
        st.warning("Dataset statistics could not be loaded. Please ensure data/processed/ contains train.csv, val.csv, and test.csv.")
    else:
        split_rows = []
        total_events = sum(stats.get(s, {}).get("rows", 0) for s in ["train", "val", "test"])

        split_meta = {
            "train": {"name": "Training Set", "file": "train.csv", "source": "KT-3 (1000) Dev Cohort", "strategy": "First 80% chronological interactions per student"},
            "val": {"name": "Validation Set", "file": "val.csv", "source": "KT-3 (1000) Dev Cohort", "strategy": "Final 20% chronological interactions per student"},
            "test": {"name": "Test Set", "file": "test.csv", "source": "KT-3 (250) Test Cohort", "strategy": "100% Unseen students (frozen holdout)"},
        }

        for s in ["train", "val", "test"]:
            s_data = stats.get(s, {})
            rows = s_data.get("rows", 0)
            cols = s_data.get("num_columns", 0)
            users = user_counts.get(s, 0)
            pct = (rows / total_events * 100) if total_events > 0 else 0.0

            split_rows.append({
                "Split": split_meta[s]["name"],
                "File": split_meta[s]["file"],
                "Source Cohort": split_meta[s]["source"],
                "Interaction Events": rows,
                "Unique Students": users,
                "Features (Cols)": cols,
                "Share of Total Events": f"{pct:.2f}%",
                "Splitting Methodology": split_meta[s]["strategy"],
            })

        df_splits = pd.DataFrame(split_rows)

        # Formatted Display Table
        st.dataframe(
            df_splits,
            column_config={
                "Interaction Events": st.column_config.NumberColumn(format="%d"),
                "Unique Students": st.column_config.NumberColumn(format="%d"),
                "Features (Cols)": st.column_config.NumberColumn(format="%d"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Visual Split Distribution
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_split_events = px.pie(
                df_splits,
                names="Split",
                values="Interaction Events",
                title="Event Volume Distribution by Split",
                hole=0.45,
                color="Split",
                color_discrete_sequence=["#3B82F6", "#F59E0B", "#10B981"]
            )
            fig_split_events.update_traces(textinfo="percent+label")
            fig_split_events.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_split_events, use_container_width=True)

        with chart_col2:
            fig_split_users = px.bar(
                df_splits,
                x="Split",
                y="Unique Students",
                text="Unique Students",
                title="Unique Students per Split Partition",
                color="Split",
                color_discrete_sequence=["#3B82F6", "#F59E0B", "#10B981"]
            )
            fig_split_users.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_split_users.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_split_users, use_container_width=True)

    st.divider()

    # -- 4, 5, 6. Formal Audit & Methodology Reports --
    st.subheader("3. Verification & Governance Reports")
    st.markdown(
        "Direct inspections of empirical audit artifacts generated during data preprocessing and validation."
    )

    report_tab1, report_tab2, report_tab3 = st.tabs([
        "Data Audit Report",
        "Leakage Report",
        "Split & Provenance Report"
    ])

    with report_tab1:
        with st.expander("Expand Empirical Data Audit Report (data_audit_report.md)", expanded=True):
            data_audit_content = get_report_markdown("data_audit_report.md")
            if data_audit_content:
                st.markdown(data_audit_content)
            else:
                st.warning("data_audit_report.md not found in reports directory.")

    with report_tab2:
        with st.expander("Expand Data Leakage Audit Report (leakage_report.md)", expanded=True):
            leakage_content = get_report_markdown("leakage_report.md")
            if leakage_content:
                st.markdown(leakage_content)
            else:
                st.warning("leakage_report.md not found in reports directory.")

    with report_tab3:
        with st.expander("Expand Dataset Splitting Report (split_report.md)", expanded=True):
            split_content = get_report_markdown("split_report.md")
            if split_content:
                st.markdown(split_content)
            else:
                st.warning("split_report.md not found in reports directory.")

    st.divider()

    # -- 7. Sample Data Preview --
    st.subheader("4. Interactive Sample Data Preview")
    st.markdown("Inspect raw processed interaction records directly from disk storage.")

    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        selected_split = st.selectbox(
            "Select Partition Split",
            options=["train", "val", "test"],
            index=0,
            format_func=lambda x: f"{x.upper()} ({split_meta.get(x, {}).get('file', x + '.csv')})" if 'split_meta' in locals() else x.upper(),
            help="Choose which CSV partition to sample from data/processed/"
        )

    with ctrl_col2:
        n_rows = st.slider(
            "Number of Records to Preview",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Slider controls the top-N rows loaded into interactive memory"
        )

    sample_df = get_dataset_sample(selected_split, n_rows)

    if sample_df is not None and not sample_df.empty:
        st.caption(
            f"Displaying top **{len(sample_df)}** records from `data/processed/{selected_split}.csv` "
            f"(Total schema: **{sample_df.shape[1]}** columns)."
        )
        st.dataframe(sample_df, use_container_width=True)
    else:
        st.warning(f"Could not load sample data for partition '{selected_split}'.")

    st.divider()

    # -- 8. Feature Overview --
    st.subheader("5. Feature Catalog & Schema Reference")
    st.markdown(
        "Complete taxonomy of engineered, normalized, and target attributes present in `train.csv`, `val.csv`, and `test.csv`."
    )

    feature_catalog = [
        {
            "Feature Name": "user_id",
            "Data Type": "string",
            "Domain Category": "Student Identity",
            "Description": "Unique identifier for the student / learner across all interaction sessions.",
            "Model Utilization": "GroupKFold grouping key, user embeddings in Model 3 (BERT-NCF)."
        },
        {
            "Feature Name": "question_id",
            "Data Type": "string",
            "Domain Category": "Item Identity",
            "Description": "Alphanumeric question item identifier from EdNet-KT3 content hierarchy.",
            "Model Utilization": "Raw item identifier; indexed into question_idx for tensor embeddings."
        },
        {
            "Feature Name": "bundle_id",
            "Data Type": "string",
            "Domain Category": "Item Identity",
            "Description": "Assessment bundle grouping identifier linking shared passages or contexts.",
            "Model Utilization": "Content context grouping; auxiliary feature for recommendation candidate retrieval."
        },
        {
            "Feature Name": "user_answer",
            "Data Type": "string (char)",
            "Domain Category": "Interaction Outcome",
            "Description": "Multiple-choice option selected by student ('a', 'b', 'c', 'd'). Quarantined from inputs.",
            "Model Utilization": "Excluded from model input tensors to prevent target leakage."
        },
        {
            "Feature Name": "correct_answer",
            "Data Type": "string (char)",
            "Domain Category": "Item Ground Truth",
            "Description": "Ground-truth correct answer option for the question item.",
            "Model Utilization": "Excluded from model input tensors to prevent target leakage."
        },
        {
            "Feature Name": "is_correct",
            "Data Type": "int (0/1)",
            "Domain Category": "Target Variable",
            "Description": "Binary correctness label: 1 if user_answer == correct_answer, 0 otherwise.",
            "Model Utilization": "Primary binary classification target for KT Models (1, 2, 3, 5)."
        },
        {
            "Feature Name": "part",
            "Data Type": "int (1-7)",
            "Domain Category": "Curriculum Section",
            "Description": "TOEIC section number (Parts 1-4: Listening; Parts 5-7: Reading).",
            "Model Utilization": "Categorical feature embedded in recurrent and transformer feature sets."
        },
        {
            "Feature Name": "tags",
            "Data Type": "string",
            "Domain Category": "Knowledge Component",
            "Description": "Semicolon or space-separated concept tag IDs indicating knowledge components tested.",
            "Model Utilization": "Multi-hot concept encoding for knowledge tracing and student mastery tracking."
        },
        {
            "Feature Name": "tag_list",
            "Data Type": "list[int]",
            "Domain Category": "Knowledge Component",
            "Description": "Parsed integer list of concept tags assigned to this specific question item.",
            "Model Utilization": "Used by knowledge state module to compute per-skill student mastery matrices."
        },
        {
            "Feature Name": "question_idx",
            "Data Type": "int",
            "Domain Category": "Embedding Index",
            "Description": "Dense zero-based integer index mapped against the 11,335 development questions.",
            "Model Utilization": "Input index for PyTorch nn.Embedding layers across all deep KT models."
        },
        {
            "Feature Name": "response_time_ms",
            "Data Type": "int / float",
            "Domain Category": "Temporal Dynamics",
            "Description": "Time taken by the student to answer the question in milliseconds (respond_ts - enter_ts).",
            "Model Utilization": "Raw engagement duration; basis for standardized response_time_norm."
        },
        {
            "Feature Name": "response_time_norm",
            "Data Type": "float",
            "Domain Category": "Temporal Dynamics",
            "Description": "Standardized response time (z-score normalized with mean and std fitted strictly on Dev).",
            "Model Utilization": "Continuous feature input in LSTM, Transformer KT, and CNN+LSTM."
        },
        {
            "Feature Name": "time_since_prev_ms",
            "Data Type": "float",
            "Domain Category": "Temporal Dynamics",
            "Description": "Elapsed interval in milliseconds since the student's preceding question response.",
            "Model Utilization": "Captures forgetting curves, spacing effects, and fatigue dynamics."
        },
        {
            "Feature Name": "time_since_prev_norm",
            "Data Type": "float",
            "Domain Category": "Temporal Dynamics",
            "Description": "Standardized inter-event time delta (z-score normalized on Dev cohort parameters).",
            "Model Utilization": "Continuous input feature modeling elapsed interval between exercises."
        },
        {
            "Feature Name": "enter_ts",
            "Data Type": "int (Unix ms)",
            "Domain Category": "Timestamp",
            "Description": "Unix timestamp in milliseconds when the question item was presented to the user.",
            "Model Utilization": "Chronological sorting and trajectory ordering verification."
        },
        {
            "Feature Name": "respond_ts",
            "Data Type": "int (Unix ms)",
            "Domain Category": "Timestamp",
            "Description": "Unix timestamp in milliseconds when the student selected their response choice.",
            "Model Utilization": "Sequence progression and temporal interval calculation."
        },
        {
            "Feature Name": "submit_ts",
            "Data Type": "float (Unix ms)",
            "Domain Category": "Timestamp",
            "Description": "Unix timestamp in milliseconds when the student committed the answer submission.",
            "Model Utilization": "Secondary submission validation and latency verification."
        },
        {
            "Feature Name": "num_responses",
            "Data Type": "int",
            "Domain Category": "Behavioral",
            "Description": "Count of attempt clicks made by the user before committing the final submission.",
            "Model Utilization": "Behavioral hesitation / deliberation feature."
        },
        {
            "Feature Name": "source",
            "Data Type": "string",
            "Domain Category": "Platform Context",
            "Description": "Origin interface mode (e.g., 'diagnosis', 'test', 'review', 'recommendation').",
            "Model Utilization": "Qualitative context indicator of learning mode."
        },
        {
            "Feature Name": "source_encoded",
            "Data Type": "int",
            "Domain Category": "Platform Context",
            "Description": "Integer label encoding of the interaction source context.",
            "Model Utilization": "Discrete categorical feature input for neural sequence models."
        },
        {
            "Feature Name": "platform",
            "Data Type": "string",
            "Domain Category": "Device Context",
            "Description": "Operating environment used by the student (e.g., 'mobile', 'web').",
            "Model Utilization": "Cross-platform behavior representation."
        },
        {
            "Feature Name": "platform_encoded",
            "Data Type": "int",
            "Domain Category": "Device Context",
            "Description": "Integer label encoding of the hardware/client platform.",
            "Model Utilization": "Categorical feature input for neural sequence models."
        },
        {
            "Feature Name": "previous_accuracy",
            "Data Type": "float [0.0 - 1.0]",
            "Domain Category": "Historical Lag",
            "Description": "Strict past-only cumulative accuracy achieved by the student prior to this interaction.",
            "Model Utilization": "Strictly past-lag feature (k < t) preventing lookahead leakage."
        },
        {
            "Feature Name": "recent_accuracy_5",
            "Data Type": "float [0.0 - 1.0]",
            "Domain Category": "Historical Lag",
            "Description": "Rolling window accuracy over the student's 5 preceding interaction attempts.",
            "Model Utilization": "Momentum feature capturing short-term student performance trends."
        },
        {
            "Feature Name": "attempt_count",
            "Data Type": "int",
            "Domain Category": "Historical Lag",
            "Description": "Cumulative count of previous questions attempted by this student prior to current step.",
            "Model Utilization": "Experience feature indicating student tenure and session maturity."
        }
    ]

    df_catalog = pd.DataFrame(feature_catalog)

    # Category Filter
    categories = ["All Categories"] + sorted(list(df_catalog["Domain Category"].unique()))
    selected_cat = st.selectbox("Filter Features by Domain Category", categories)

    if selected_cat != "All Categories":
        filtered_catalog = df_catalog[df_catalog["Domain Category"] == selected_cat]
    else:
        filtered_catalog = df_catalog

    st.dataframe(
        filtered_catalog,
        column_config={
            "Feature Name": st.column_config.TextColumn("Feature Name", width="medium"),
            "Data Type": st.column_config.TextColumn("Data Type", width="small"),
            "Domain Category": st.column_config.TextColumn("Category", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Model Utilization": st.column_config.TextColumn("Model Utilization", width="large"),
        },
        use_container_width=True,
        hide_index=True
    )
