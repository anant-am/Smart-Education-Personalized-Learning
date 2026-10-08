"""
Smart Education Project — Video + Audio Attention Analysis Page
=================================================================
Interactive Streamlit interface for:
1. Video & audio engagement analysis (.mp4, .avi, .mov, .mkv)
2. Multimodal feature inspection (Facial gaze, motion, acoustics)
3. Attention classification (Attentive, Partially Attentive, Inattentive)
4. Knowledge state and attention-adaptive recommendation synthesis
"""

import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_inference import AttentionInferenceEngine
from attention.attention_pipeline import run_video_attention_pipeline, adjust_recommendations_by_attention
from scripts.run_video_attention import create_synthetic_demo_video


@st.cache_resource(show_spinner="Loading Attention Inference Engine into memory...")
def get_cached_attention_engine():
    """Singleton-cached attention engine."""
    return AttentionInferenceEngine()


def render(engine):
    st.markdown('<div class="main-header">🎥 Video + Audio Attention Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Multimodal engagement modeling, facial/acoustic telemetry, and attention-adaptive personalized learning.'
        '</div>',
        unsafe_allow_html=True
    )

    try:
        att_engine = get_cached_attention_engine()
    except Exception as e:
        st.error(f"Failed to load Attention Engine: {str(e)}")
        att_engine = None

    # ── Controls Section ──
    col_input1, col_input2 = st.columns([3, 2])

    with col_input1:
        st.subheader("1. Video Source Input")
        upload_mode = st.radio(
            "Select Video Input Mode",
            ["Upload Video File", "Use Synthetic Demonstration Video"],
            horizontal=True
        )

        video_path = None
        if upload_mode == "Upload Video File":
            uploaded_file = st.file_uploader("Upload Learner Video (.mp4, .avi, .mov, .mkv)", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                # Save to temp file
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                    tf.write(uploaded_file.read())
                    video_path = tf.name
        else:
            st.info("Demonstration mode: generates a real-time synthetic learner session with visual facial cues.")
            if st.button("Generate & Load Demo Video"):
                with st.spinner("Rendering demo video..."):
                    video_path = create_synthetic_demo_video(duration_sec=3)
                    st.session_state["demo_video_path"] = video_path

            if "demo_video_path" in st.session_state and os.path.exists(st.session_state["demo_video_path"]):
                video_path = st.session_state["demo_video_path"]

    with col_input2:
        st.subheader("2. Student Profile Integration")
        student_id = st.text_input("Enter Student ID (e.g. u1165, u502)", value="u1165").strip()
        st.caption("Links diagnosed attention state with authentic EdNet learner history, concept mastery, and remedial recommendations.")

        run_button = st.button("🚀 Analyze Video & Predict Attention", type="primary", use_container_width=True)

    if not run_button and "last_attention_result" not in st.session_state:
        st.info("Upload a video or generate a demo video, then click **Analyze Video & Predict Attention**.")
        return

    if run_button:
        if not video_path or not os.path.exists(video_path):
            st.warning("Please provide or generate a valid video file first.")
            return

        with st.spinner("Extracting visual & acoustic features and executing multimodal model forward pass..."):
            pipeline_res = run_video_attention_pipeline(
                video_path=video_path,
                student_id=student_id if student_id else None,
                attention_engine=att_engine,
                kt_engine=engine
            )
            st.session_state["last_attention_result"] = pipeline_res
            st.session_state["last_video_path"] = video_path

    res = st.session_state.get("last_attention_result")
    active_video = st.session_state.get("last_video_path")

    if not res:
        return

    att = res.get("attention", {})
    att_class = att.get("attention_class", "Unknown")
    confidence = att.get("confidence", 0.0)
    probs = att.get("probabilities", {})

    st.markdown("---")

    # ── Overview Metrics ──
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    # Class color styling
    class_colors = {
        "Attentive": "#16a34a",
        "Partially Attentive": "#ca8a04",
        "Inattentive": "#dc2626"
    }
    badge_color = class_colors.get(att_class, "#2563eb")

    with m_col1:
        st.markdown(f"**Attention State**<br><span style='font-size:1.4rem; font-weight:700; color:{badge_color};'>{att_class}</span>", unsafe_allow_html=True)
    with m_col2:
        st.metric("Prediction Confidence", f"{confidence*100:.1f}%")
    with m_col3:
        v_status = "✅ Active" if att.get("video_available") else "⚠️ Unavailable"
        st.metric("Video Stream", v_status)
    with m_col4:
        a_status = "✅ Active" if att.get("audio_available") else "⚠️ Unavailable"
        st.metric("Audio Stream", a_status)

    # ── Video & Chart Display ──
    v_col, c_col = st.columns([1, 1])

    with v_col:
        st.markdown("#### Learner Video Stream")
        if active_video and os.path.exists(active_video):
            st.video(active_video)
        else:
            st.caption("Video playback unavailable.")

        if att.get("warnings"):
            with st.expander("Diagnostic Telemetry Warnings"):
                for w in att["warnings"]:
                    st.caption(f"• {w}")

    with c_col:
        st.markdown("#### Attention Probability Distribution")
        categories = list(probs.keys())
        values = [probs[c] * 100 for c in categories]
        colors = [class_colors.get(c, "#3b82f6") for c in categories]

        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="auto"
            )
        ])
        fig.update_layout(
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Extracted Telemetry Table ──
    with st.expander("🔍 Detailed Extracted Feature Telemetry (10-D Video + 10-D Audio)", expanded=False):
        v_feats = att.get("video_features", {})
        a_feats = att.get("audio_features", {})

        tf_col1, tf_col2 = st.columns(2)
        with tf_col1:
            st.markdown("**Visual Features (OpenCV Extraction)**")
            if v_feats:
                v_df = pd.DataFrame([{"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in v_feats.items()])
                st.dataframe(v_df, hide_index=True, use_container_width=True)
        with tf_col2:
            st.markdown("**Acoustic Features (SciPy / SoundFile Extraction)**")
            if a_feats:
                a_df = pd.DataFrame([{"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in a_feats.items()])
                st.dataframe(a_df, hide_index=True, use_container_width=True)

    # ── Personalized Recommendations Integration ──
    if res.get("student_id"):
        st.markdown("---")
        st.subheader(f"🧠 Attention-Adaptive Personalized Interventions (Student {res['student_id']})")

        pers = res.get("personalization", {})
        strat_title = pers.get("strategy_title", "Standard Recommendation")
        rationale = pers.get("rationale", "")

        st.info(f"**Adaptive Pedagogical Strategy:** {strat_title}\n\n{rationale}")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("**1. Attention-Adapted Learning Path**")
            adapted_recs = pers.get("adapted_recommendations", [])
            if adapted_recs:
                for idx, r in enumerate(adapted_recs[:5], 1):
                    rtype = r.get("type", "Resource").upper()
                    rid = r.get("id", "N/A")
                    cid = r.get("concept", "N/A")
                    pacing = f" • *{r.get('pacing_tag')}*" if "pacing_tag" in r else ""
                    dur = f" ({r.get('duration_sec')}s)" if r.get("duration_sec") else ""
                    st.markdown(f"**{idx}. [{rtype}]** ID: `{rid}` | Concept: **#{cid}** {dur}{pacing}")
            else:
                st.caption("No recommendations generated.")

        with rec_col2:
            st.markdown("**2. Diagnosed Learning Gaps Targeted**")
            gaps = res.get("learning_gaps", [])
            if gaps:
                gap_data = []
                for g in gaps[:5]:
                    gap_data.append({
                        "Concept ID": f"#{g.get('concept_id', g.get('concept'))}",
                        "Mastery": f"{g.get('mastery', 0.0)*100:.1f}%",
                        "Deficit": f"{g.get('gap_severity', 0.0)*100:.1f}%",
                        "Attempts": g.get("attempts", 1)
                    })
                st.dataframe(pd.DataFrame(gap_data), hide_index=True, use_container_width=True)
            else:
                st.success("No critical learning deficits below 60% mastery threshold detected!")

        # ── AI Explanation (Ollama Integration) ──
        st.markdown("---")
        with st.expander("🤖 AI Multimodal Learning Explanation", expanded=True):
            st.caption("Generative AI combines real-time attention signals with concept knowledge state to explain pacing and cognitive focus.")
            if st.button("✨ Generate AI Explanation", key="btn_att_ai_exp"):
                with st.spinner("Analyzing multimodal telemetry with Ollama..."):
                    from ollama_ai.learner_context import build_learner_context
                    from ollama_ai.response_generator import ResponseGenerator
                    ai_ctx = build_learner_context(inference_result=res, attention_result=res, student_id=res.get("student_id"))
                    ai_gen = ResponseGenerator()
                    st.session_state["att_ai_result"] = ai_gen.explain_multimodal_attention(ai_ctx)

            if "att_ai_result" in st.session_state:
                att_exp = st.session_state["att_ai_result"]
                badge = "🟢 Ollama AI" if att_exp.get("ai_available") else "⚪ Deterministic Pedagogical Fallback (Ollama Offline)"
                st.markdown(f"**Interpretation Source:** `{badge}`")
                st.markdown(att_exp.get("explanation", ""))

        st.caption(f"⚡ End-to-end multimodal pipeline latency: {res.get('pipeline_execution_time_ms', 0):.1f} ms")

