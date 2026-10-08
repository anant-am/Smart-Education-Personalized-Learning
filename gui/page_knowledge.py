"""
Smart Education -- Knowledge State Visualization Page
=====================================================
Interactive knowledge state dashboard visualizing model-derived mastery
scores alongside historical baseline accuracies for individual learners.
Supports configurable mastery thresholds, deficit highlighting in red,
KPI summary metrics, and styled learning gap analysis tables.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.inference_service import predict_student


def _extract_knowledge_state_dataframe(k_state: Any) -> pd.DataFrame:
    """Safely parses knowledge_state into a normalized DataFrame."""
    rows: List[Dict[str, Any]] = []
    if not k_state:
        return pd.DataFrame(rows)

    if isinstance(k_state, dict):
        for k, v in k_state.items():
            if isinstance(v, dict):
                cid = v.get("concept_id", k)
                cname = v.get("concept_name", f"Concept_{cid}")
                score = float(v.get("mastery_score", 0.0))
            else:
                cid = k
                cname = f"Concept_{cid}"
                score = float(v)
            rows.append({
                "concept_id": str(cid),
                "concept_name": str(cname),
                "mastery_score": score,
            })
    elif isinstance(k_state, list):
        for item in k_state:
            if isinstance(item, dict):
                cid = item.get("concept_id", "")
                cname = item.get("concept_name", f"Concept_{cid}")
                score = float(item.get("mastery_score", 0.0))
                rows.append({
                    "concept_id": str(cid),
                    "concept_name": str(cname),
                    "mastery_score": score,
                })

    return pd.DataFrame(rows)


def _extract_baseline_state_dataframe(b_state: Any) -> pd.DataFrame:
    """Safely parses baseline_state into a normalized DataFrame."""
    rows: List[Dict[str, Any]] = []
    if not b_state:
        return pd.DataFrame(rows)

    if isinstance(b_state, dict):
        for k, v in b_state.items():
            if isinstance(v, dict):
                cid = v.get("concept_id", k)
                cname = v.get("concept_name", f"Concept_{cid}")
                acc = float(v.get("accuracy", 0.0))
                attempts = int(v.get("total_attempts", 0))
                correct = int(v.get("correct_count", 0))
            else:
                cid = k
                cname = f"Concept_{cid}"
                acc = float(v)
                attempts = 1
                correct = int(v >= 0.5)
            rows.append({
                "concept_id": str(cid),
                "concept_name": str(cname),
                "accuracy": acc,
                "total_attempts": attempts,
                "correct_count": correct,
            })
    elif isinstance(b_state, list):
        for item in b_state:
            if isinstance(item, dict):
                cid = item.get("concept_id", "")
                cname = item.get("concept_name", f"Concept_{cid}")
                acc = float(item.get("accuracy", 0.0))
                attempts = int(item.get("total_attempts", 0))
                correct = int(item.get("correct_count", 0))
                rows.append({
                    "concept_id": str(cid),
                    "concept_name": str(cname),
                    "accuracy": acc,
                    "total_attempts": attempts,
                    "correct_count": correct,
                })

    return pd.DataFrame(rows)


def render(engine: Optional[Any] = None) -> None:
    """
    Entry point for the Knowledge State Visualization page.

    Args:
        engine: MultiModelInferenceEngine instance or None.
    """
    st.title("Knowledge State Visualization")

    result = st.session_state.get("last_prediction")

    # If no prediction exists, show prompt message
    if not result:
        st.info("Run a prediction in Student Analysis first.")

        # Convenient sample loading button if engine is available
        if engine is not None:
            st.markdown("---")
            st.markdown("##### Quick Demo")
            st.caption("You can run a sample inference for demonstration directly:")
            if st.button("Load Sample Student (u1165)", key="btn_load_sample_u1165"):
                with st.spinner("Running multi-model inference for student u1165..."):
                    sample_res = predict_student(engine, "u1165")
                    if sample_res:
                        st.session_state["last_prediction"] = sample_res
                        st.rerun()
                    else:
                        st.error("Could not load sample student u1165.")
        return

    # Extract student identifier and metadata
    student_id = str(result.get("student_id", "Unknown Student"))
    split_info = str(result.get("split", "")).upper()
    split_badge = f" [{split_info} SET]" if split_info else ""
    latency = result.get("inference_time_ms", 0.0)
    total_events = result.get("total_events", 0)

    # 1. Header with student ID
    st.markdown(f"### Student Profile: **{student_id}**{split_badge}")
    st.caption(f"Historical Interactions: {total_events:,} | Latency: {latency:.2f} ms")

    # 3. Mastery threshold slider (0.0 to 1.0, default 0.60)
    st.markdown("---")
    col_ctrl1, col_ctrl2 = st.columns([3, 2])
    with col_ctrl1:
        threshold = st.slider(
            "Mastery Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
            format="%.2f",
            help="Concepts with mastery score or accuracy below this threshold are marked as learning gaps (red)."
        )
    with col_ctrl2:
        sort_choice = st.selectbox(
            "Sort Order for Charts",
            options=["Deficit First (Ascending)", "Mastery First (Descending)", "Concept Identifier"],
            index=0,
            help="Order in which concepts are displayed along the horizontal axis."
        )

    # Extract datasets
    raw_k_state = result.get("knowledge_state", {})
    raw_b_state = result.get("baseline_state", {})
    df_k = _extract_knowledge_state_dataframe(raw_k_state)
    df_b = _extract_baseline_state_dataframe(raw_b_state)

    # 5. Summary metrics: total concepts tracked, concepts above threshold, concepts below threshold
    if not df_k.empty:
        total_tracked = len(df_k)
        above_count = int((df_k["mastery_score"] >= threshold).sum())
        below_count = int((df_k["mastery_score"] < threshold).sum())
    elif not df_b.empty:
        total_tracked = len(df_b)
        above_count = int((df_b["accuracy"] >= threshold).sum())
        below_count = int((df_b["accuracy"] < threshold).sum())
    else:
        total_tracked = 0
        above_count = 0
        below_count = 0

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Concepts Tracked", total_tracked)
    col_m2.metric("Concepts Above Threshold", above_count)
    col_m3.metric(
        "Concepts Below Threshold",
        below_count,
        delta=f"-{below_count} deficit(s)" if below_count > 0 else "All Mastered",
        delta_color="inverse" if below_count > 0 else "normal"
    )

    st.markdown("---")

    # 2. Two-column layout:
    #    - Left: 'Model-Derived Knowledge State' - bar chart of mastery scores from result['knowledge_state']
    #    - Right: 'Historical Baseline' - bar chart from result.get('baseline_state', {}) showing accuracy values
    col_left, col_right = st.columns(2)

    # --- LEFT COLUMN: Model-Derived Knowledge State ---
    with col_left:
        st.subheader("Model-Derived Knowledge State")
        if df_k.empty:
            st.warning("No model-derived knowledge state available for this prediction.")
        else:
            df_k_plot = df_k.copy()
            if sort_choice == "Deficit First (Ascending)":
                df_k_plot = df_k_plot.sort_values("mastery_score", ascending=True)
            elif sort_choice == "Mastery First (Descending)":
                df_k_plot = df_k_plot.sort_values("mastery_score", ascending=False)
            else:
                df_k_plot["cid_num"] = pd.to_numeric(df_k_plot["concept_id"], errors="coerce").fillna(999999)
                df_k_plot = df_k_plot.sort_values("cid_num", ascending=True)

            # Highlight concepts below threshold in red (#EF4444)
            colors_k = [
                "#EF4444" if score < threshold else "#10B981"
                for score in df_k_plot["mastery_score"]
            ]

            fig_k = go.Figure(data=[
                go.Bar(
                    x=df_k_plot["concept_name"],
                    y=df_k_plot["mastery_score"],
                    marker_color=colors_k,
                    text=[f"{s:.1%}" for s in df_k_plot["mastery_score"]],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Mastery Score: %{y:.3f} (%{text})<extra></extra>"
                )
            ])
            fig_k.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#DC2626",
                line_width=2,
                annotation_text=f"Threshold ({threshold:.2f})",
                annotation_position="top right"
            )
            fig_k.update_layout(
                title=dict(text="Deep KT Model Mastery Scores", font=dict(size=14)),
                xaxis_title="Concept",
                yaxis_title="Mastery Score",
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                xaxis=dict(tickangle=-45),
                template="plotly_white",
                height=420,
                margin=dict(l=20, r=20, t=50, b=80),
            )
            st.plotly_chart(fig_k, use_container_width=True)
            st.caption(f"Legend: Red = Below Threshold (< {threshold:.2f}) | Green = Mastered (>= {threshold:.2f})")

    # --- RIGHT COLUMN: Historical Baseline ---
    with col_right:
        st.subheader("Historical Baseline")
        if df_b.empty:
            st.info("No historical baseline data available. Baseline accuracy is computed from historical interaction records in dataset splits.")
        else:
            df_b_plot = df_b.copy()
            if sort_choice == "Deficit First (Ascending)":
                df_b_plot = df_b_plot.sort_values("accuracy", ascending=True)
            elif sort_choice == "Mastery First (Descending)":
                df_b_plot = df_b_plot.sort_values("accuracy", ascending=False)
            else:
                df_b_plot["cid_num"] = pd.to_numeric(df_b_plot["concept_id"], errors="coerce").fillna(999999)
                df_b_plot = df_b_plot.sort_values("cid_num", ascending=True)

            # Highlight concepts below threshold in red (#EF4444)
            colors_b = [
                "#EF4444" if acc < threshold else "#3B82F6"
                for acc in df_b_plot["accuracy"]
            ]

            fig_b = go.Figure(data=[
                go.Bar(
                    x=df_b_plot["concept_name"],
                    y=df_b_plot["accuracy"],
                    marker_color=colors_b,
                    customdata=df_b_plot[["total_attempts", "correct_count"]].values,
                    text=[f"{a:.1%}" for a in df_b_plot["accuracy"]],
                    textposition="auto",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Historical Accuracy: %{y:.3f} (%{text})<br>"
                        "Total Attempts: %{customdata[0]}<br>"
                        "Correct Answers: %{customdata[1]}<extra></extra>"
                    )
                )
            ])
            fig_b.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#DC2626",
                line_width=2,
                annotation_text=f"Threshold ({threshold:.2f})",
                annotation_position="top right"
            )
            fig_b.update_layout(
                title=dict(text="Empirical Historical Accuracy", font=dict(size=14)),
                xaxis_title="Concept",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                xaxis=dict(tickangle=-45),
                template="plotly_white",
                height=420,
                margin=dict(l=20, r=20, t=50, b=80),
            )
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption(f"Legend: Red = Below Threshold (< {threshold:.2f}) | Blue = Historical Baseline (>= {threshold:.2f})")

    st.markdown("---")

    # 4. Learning gaps table from result['learning_gaps'] as a styled dataframe
    st.subheader("Learning Gaps Analysis")

    raw_gaps = result.get("learning_gaps", [])

    # If the user adjusts the threshold slider from default 0.60, evaluate gaps for chosen threshold
    if abs(threshold - 0.60) < 1e-4 and raw_gaps:
        active_gaps = list(raw_gaps)
    else:
        # Dynamically compute gaps below current threshold from knowledge_state (or baseline_state)
        active_gaps = []
        source_state = raw_k_state if raw_k_state else raw_b_state
        if isinstance(source_state, dict):
            for cid, info in source_state.items():
                if isinstance(info, dict):
                    score = float(info.get("mastery_score", info.get("accuracy", 0.0)))
                    attempts = int(info.get("total_attempts", info.get("predicted_observations", 1)))
                    recent_acc = float(info.get("recent_accuracy", score))
                    cname = info.get("concept_name", f"Concept_{cid}")
                else:
                    score = float(info)
                    attempts = 1
                    recent_acc = score
                    cname = f"Concept_{cid}"

                if score < threshold:
                    active_gaps.append({
                        "concept_id": cid,
                        "concept_name": cname,
                        "mastery": score,
                        "attempts": attempts,
                        "recent_accuracy": recent_acc,
                        "gap_severity": float(threshold - score),
                    })
        active_gaps.sort(key=lambda x: x.get("mastery", 0.0))

    if not active_gaps:
        st.success(f"No learning gaps identified below the mastery threshold ({threshold:.2f}). Student has demonstrated satisfactory proficiency across evaluated concepts.")
    else:
        df_gaps = pd.DataFrame(active_gaps)

        # Enforce numeric types
        if "mastery" in df_gaps.columns:
            df_gaps["mastery"] = pd.to_numeric(df_gaps["mastery"], errors="coerce").fillna(0.0)
        if "gap_severity" in df_gaps.columns:
            df_gaps["gap_severity"] = pd.to_numeric(df_gaps["gap_severity"], errors="coerce").fillna(0.0)
        if "attempts" in df_gaps.columns:
            df_gaps["attempts"] = pd.to_numeric(df_gaps["attempts"], errors="coerce").fillna(1).astype(int)
        if "recent_accuracy" in df_gaps.columns:
            df_gaps["recent_accuracy"] = pd.to_numeric(df_gaps["recent_accuracy"], errors="coerce").fillna(0.0)

        # Rename columns for presentation
        rename_map = {
            "concept_id": "Concept ID",
            "concept_name": "Concept Name",
            "mastery": "Mastery Score",
            "attempts": "Attempts",
            "recent_accuracy": "Recent Accuracy",
            "gap_severity": "Gap Severity",
        }
        df_display = df_gaps.rename(columns=rename_map)

        # Setup styling
        formatters = {}
        if "Mastery Score" in df_display.columns:
            formatters["Mastery Score"] = "{:.1%}"
        if "Recent Accuracy" in df_display.columns:
            formatters["Recent Accuracy"] = "{:.1%}"
        if "Gap Severity" in df_display.columns:
            formatters["Gap Severity"] = "{:.3f}"
        if "Attempts" in df_display.columns:
            formatters["Attempts"] = "{:d}"

        styled_df = df_display.style.format(formatters)

        if "Gap Severity" in df_display.columns and len(df_display) > 0:
            styled_df = styled_df.background_gradient(subset=["Gap Severity"], cmap="Reds", vmin=0.0)

        if "Mastery Score" in df_display.columns and len(df_display) > 0:
            styled_df = styled_df.background_gradient(subset=["Mastery Score"], cmap="RdYlGn", vmin=0.0, vmax=1.0)

        st.warning(f"Identified {len(df_gaps)} concept gap(s) requiring targeted intervention (Threshold: {threshold:.2f}).")
        st.dataframe(styled_df, use_container_width=True)

        # CSV Download for educational report export
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Learning Gaps Report (CSV)",
            data=csv_data,
            file_name=f"learning_gaps_{student_id}.csv",
            mime="text/csv",
            key="btn_download_gaps_csv"
        )

        # ── AI Explanation of Learning Gaps (Ollama Integration) ──
        with st.expander("🤖 AI Learning Gap Diagnostics (Ollama)", expanded=False):
            st.caption("Generative AI interprets diagnosed concept deficits and explains why they represent foundational priorities.")
            if st.button("🧠 Explain My Learning Gaps with AI", key="btn_explain_gaps_ai"):
                with st.spinner("Generating diagnostic explanation with Ollama..."):
                    from ollama_ai.learner_context import build_learner_context
                    from ollama_ai.response_generator import ResponseGenerator
                    ai_ctx = build_learner_context(inference_result=result, student_id=student_id)
                    ai_gen = ResponseGenerator()
                    st.session_state["gaps_ai_explanation"] = ai_gen.explain_learning_gaps(ai_ctx)

            if "gaps_ai_explanation" in st.session_state:
                exp_data = st.session_state["gaps_ai_explanation"]
                badge = "🟢 Ollama AI" if exp_data.get("ai_available") else "⚪ Deterministic Pedagogical Fallback (Ollama Offline)"
                st.markdown(f"**Interpretation Source:** `{badge}`")
                st.markdown(exp_data.get("explanation", ""))

    # Optional detailed comparison expander if both states are present

    if not df_k.empty and not df_b.empty:
        with st.expander("Detailed Comparison: Model-Derived Mastery vs. Historical Accuracy"):
            merged = pd.merge(
                df_k[["concept_id", "concept_name", "mastery_score"]],
                df_b[["concept_id", "accuracy", "total_attempts", "correct_count"]],
                on="concept_id",
                how="outer"
            )
            # Combine concept_name if missing from one side
            merged["Concept Name"] = merged["concept_name"].fillna(merged["concept_id"].apply(lambda x: f"Concept_{x}"))
            merged["Model Mastery"] = merged["mastery_score"].fillna(0.0)
            merged["Historical Accuracy"] = merged["accuracy"].fillna(0.0)
            merged["Delta (Model - History)"] = merged["Model Mastery"] - merged["Historical Accuracy"]
            merged["Total Attempts"] = merged["total_attempts"].fillna(0).astype(int)
            merged["Correct Count"] = merged["correct_count"].fillna(0).astype(int)

            comp_df = merged[[
                "concept_id", "Concept Name", "Model Mastery",
                "Historical Accuracy", "Delta (Model - History)", "Total Attempts", "Correct Count"
            ]].rename(columns={"concept_id": "Concept ID"})

            comp_styled = comp_df.style.format({
                "Model Mastery": "{:.1%}",
                "Historical Accuracy": "{:.1%}",
                "Delta (Model - History)": "{:+.1%}",
                "Total Attempts": "{:d}",
                "Correct Count": "{:d}"
            }).background_gradient(subset=["Delta (Model - History)"], cmap="coolwarm", vmin=-0.5, vmax=0.5)

            st.dataframe(comp_styled, use_container_width=True)
