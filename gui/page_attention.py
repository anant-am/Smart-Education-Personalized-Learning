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
            [
                "Feature Vectors (Direct Model Inference)",
                "Upload Video File",
                "Use Synthetic Demonstration Video"
            ],
            horizontal=False
        )

        video_path = None
        v_feat = None
        a_feat = None

        video_stream_active = True
        audio_stream_active = True

        if upload_mode == "Feature Vectors (Direct Model Inference)":
            st.info("Direct inference using trained attention checkpoint & scalers on authentic 10-D multimodal telemetry.")

            stream_col1, stream_col2 = st.columns(2)
            with stream_col1:
                video_stream_active = st.checkbox("Visual Stream Active", value=True, help="Toggle facial/gaze sensor stream")
            with stream_col2:
                audio_stream_active = st.checkbox("Acoustic Stream Active", value=True, help="Toggle vocal/speech sensor stream")

            video_csv_path = PROJECT_ROOT / "video audio dataset" / "video_attention_dataset.csv"
            audio_csv_path = PROJECT_ROOT / "video audio dataset" / "audio_attention_dataset.csv"

            telemetry_sources = ["Authentic Learner Records (video audio dataset)", "Diagnostic Archetype Presets"]
            telemetry_choice = st.radio("Telemetry Data Source", telemetry_sources, horizontal=True)

            if telemetry_choice == "Authentic Learner Records (video audio dataset)" and video_csv_path.exists() and audio_csv_path.exists():
                try:
                    df_v = pd.read_csv(video_csv_path)
                    df_a = pd.read_csv(audio_csv_path)

                    v_cols = ['Face_Visibility', 'Eye_Gaze_Score', 'Head_Pose_Stability', 'Blink_Rate', 'Facial_Expression_Intensity', 'Body_Posture_Score', 'Movement_Level', 'Focus_Duration', 'Screen_Orientation', 'Attention_Score']
                    a_cols = ['Speech_Presence', 'Voice_Energy', 'Speech_Clarity', 'Background_Noise_Level', 'Speaking_Rate', 'Pause_Frequency', 'Response_Delay', 'Interaction_Frequency', 'Audio_Engagement_Score', 'Listening_Consistency']

                    sample_labels = [
                        f"Record #{i+1}: {df_v.iloc[i].get('General_Class', 'Unknown')} (Gaze: {df_v.iloc[i]['Eye_Gaze_Score']:.2f}, Voice: {df_a.iloc[i]['Voice_Energy']:.2f})"
                        for i in range(min(50, len(df_v)))
                    ]
                    selected_sample_idx = st.selectbox(
                        "Select from 2,000 Authentic Multimodal Learner Records:",
                        options=list(range(len(sample_labels))),
                        format_func=lambda idx: sample_labels[idx]
                    )

                    v_feat = df_v.iloc[selected_sample_idx][v_cols].to_numpy(dtype=np.float32)
                    a_feat = df_a.iloc[selected_sample_idx][a_cols].to_numpy(dtype=np.float32)
                    ground_truth = df_v.iloc[selected_sample_idx].get('General_Class', 'N/A')
                    st.caption(f"Ground Truth Class in Dataset: **{ground_truth}** (Sample Index: {selected_sample_idx})")
                except Exception as ex:
                    st.warning(f"Could not parse authentic records: {ex}. Using presets.")
                    telemetry_choice = "Diagnostic Archetype Presets"

            if telemetry_choice == "Diagnostic Archetype Presets" or v_feat is None:
                profile_choice = st.selectbox(
                    "Select Learner Profile Preset",
                    [
                        "Attentive Student Profile (High gaze, steady posture, active speech)",
                        "Partially Attentive Profile (Moderate focus, occasional glance away)",
                        "Inattentive Student Profile (Low gaze, high movement, low speech engagement)"
                    ]
                )
                if "Attentive Student" in profile_choice:
                    v_feat = np.array([0.98, 0.92, 0.88, 16.5, 0.75, 0.90, 0.20, 240.0, 0.95, 0.92], dtype=np.float32)
                    a_feat = np.array([1.0, 0.85, 0.90, 0.12, 3.2, 0.25, 0.8, 0.85, 0.92, 0.95], dtype=np.float32)
                elif "Partially Attentive" in profile_choice:
                    v_feat = np.array([0.70, 0.55, 0.50, 22.0, 0.45, 0.60, 0.48, 120.0, 0.70, 0.55], dtype=np.float32)
                    a_feat = np.array([1.0, 0.50, 0.60, 0.35, 2.4, 0.55, 2.1, 0.45, 0.52, 0.60], dtype=np.float32)
                else:
                    v_feat = np.array([0.25, 0.18, 0.22, 28.5, 0.20, 0.30, 0.82, 35.0, 0.30, 0.22], dtype=np.float32)
                    a_feat = np.array([0.0, 0.15, 0.30, 0.75, 0.8, 0.85, 4.8, 0.10, 0.18, 0.25], dtype=np.float32)

        elif upload_mode == "Upload Video File":
            st.info("📁 Raw Video Mode: Extracts authentic 10-D facial telemetry and acoustic speech signals directly from your uploaded media container.")
            uploaded_file = st.file_uploader("Upload Learner Video (.mp4, .avi, .mov, .mkv)", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                # Save to persistent temp file and keep in session state across reruns
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                    uploaded_file.seek(0)
                    tf.write(uploaded_file.read())
                    video_path = tf.name
                    st.session_state["uploaded_video_path"] = video_path
            elif "uploaded_video_path" in st.session_state and os.path.exists(st.session_state["uploaded_video_path"]):
                video_path = st.session_state["uploaded_video_path"]
        else:
            st.info("🧪 Synthetic Demonstration Mode: Generates a test video with animated facial cues to demonstrate the video-decoding and feature-extraction pipeline. (Explicitly labelled as a synthetic demonstration; not a real student recording).")
            if st.button("Generate & Load Demo Video"):
                try:
                    with st.spinner("Rendering demo video..."):
                        video_path = create_synthetic_demo_video(duration_sec=3)
                        st.session_state["demo_video_path"] = video_path
                        st.success("Demo video rendered successfully.")
                except ImportError as ie:
                    st.warning(f"⚠️ {str(ie)}")
                except Exception as e:
                    st.error(f"Error rendering demo video: {str(e)}")

            if "demo_video_path" in st.session_state and os.path.exists(st.session_state["demo_video_path"]):
                video_path = st.session_state["demo_video_path"]

    with col_input2:
        st.subheader("2. Student Profile Integration")
        student_id = st.text_input("Enter Student ID (e.g. u1165, u502)", value="u1165").strip()
        st.caption("Links diagnosed attention state with authentic EdNet learner history, concept mastery, and remedial recommendations.")

        run_button = st.button("🚀 Analyze & Predict Attention", type="primary", use_container_width=True)

    if not run_button and "last_attention_result" not in st.session_state:
        st.info("Select an input mode and click **Analyze & Predict Attention**.")
        return

    if run_button:
        if upload_mode == "Feature Vectors (Direct Model Inference)":
            if v_feat is None or a_feat is None:
                st.warning("Please select a valid feature vector profile.")
                return
            with st.spinner("Executing multimodal attention model forward pass with checkpoint..."):
                pipeline_res = run_video_attention_pipeline(
                    video_features=v_feat,
                    audio_features=a_feat,
                    student_id=student_id if student_id else None,
                    attention_engine=att_engine,
                    kt_engine=engine,
                    video_available=video_stream_active,
                    audio_available=audio_stream_active
                )
                st.session_state["last_attention_result"] = pipeline_res
                st.session_state["last_video_path"] = None
        else:
            active_vid = video_path
            if not active_vid and upload_mode == "Upload Video File":
                active_vid = st.session_state.get("uploaded_video_path")
            elif not active_vid and upload_mode == "Use Synthetic Demonstration Video":
                active_vid = st.session_state.get("demo_video_path")

            if not active_vid or not os.path.exists(active_vid):
                st.warning("Please upload a video or click 'Generate & Load Demo Video' first, or select 'Feature Vectors' mode.")
                return

            with st.spinner("Extracting visual & acoustic features and executing multimodal model forward pass..."):
                pipeline_res = run_video_attention_pipeline(
                    video_path=active_vid,
                    student_id=student_id if student_id else None,
                    attention_engine=att_engine,
                    kt_engine=engine
                )
                st.session_state["last_attention_result"] = pipeline_res
                st.session_state["last_video_path"] = active_vid

    res = st.session_state.get("last_attention_result")
    active_video = st.session_state.get("last_video_path")

    if not res:
        return

    att = res.get("attention", {})
    att_class = att.get("attention_class", "Unknown")
    confidence = att.get("confidence", 0.0)
    probs = att.get("probabilities", {})

    st.markdown("---")

    # ── Check for Extraction Failure vs Model Prediction ──
    is_extraction_failure = (
        att.get("status") == "Extraction Failure"
        or att_class == "Extraction Failed"
        or (not att.get("video_available") and not att.get("audio_available"))
    )

    if is_extraction_failure:
        st.error(
            f"❌ **Feature Extraction Failed**: {att.get('error', 'Neither visual face cues nor acoustic speech could be extracted.')}\n\n"
            "The model requires at least one active modality stream to perform inference. Ensure the video clearly frames the learner's face under adequate lighting, or includes an active audio track with vocal speech."
        )

    # ── Overview Metrics ──
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    # Class color styling
    class_colors = {
        "Attentive": "#16a34a",
        "Partially Attentive": "#ca8a04",
        "Inattentive": "#dc2626"
    }
    badge_color = class_colors.get(att_class, "#6b7280" if is_extraction_failure else "#2563eb")
    display_class = "Extraction Failed" if is_extraction_failure else att_class

    with m_col1:
        st.markdown(f"**Attention State**<br><span style='font-size:1.4rem; font-weight:700; color:{badge_color};'>{display_class}</span>", unsafe_allow_html=True)
    with m_col2:
        conf_text = "N/A" if is_extraction_failure else f"{confidence*100:.1f}%"
        st.metric("Prediction Confidence", conf_text)
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
            st.caption("ℹ️ Video playback confirms container decoding; features are extracted from sampled frames.")
        else:
            st.caption("Direct feature vector evaluation (no video stream).")

        if att.get("warnings"):
            with st.expander("Diagnostic Telemetry Warnings", expanded=is_extraction_failure):
                for w in att["warnings"]:
                    st.caption(f"• {w}")

    with c_col:
        st.markdown("#### Attention Probability Distribution")
        if not is_extraction_failure and probs and any(v > 0 for v in probs.values()):
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
        else:
            st.info("⚠️ Probability distribution unavailable: No valid predictions were computed because stream extraction did not detect face or voice features.")

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

        pers = res.get("personalization") or {}
        strat_title = pers.get("strategy_title", "Standard Recommendation")
        rationale = pers.get("rationale", "Attention state diagnosed. Direct multimodal feature inference active.")

        st.info(f"**Adaptive Pedagogical Strategy:** {strat_title}\n\n{rationale}")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("**1. Attention-Adapted Learning Path**")
            adapted_recs = pers.get("adapted_recommendations") or pers.get("adjusted_recommendations") or res.get("recommendations", [])
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
            gaps = res.get("learning_gaps") or []
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

