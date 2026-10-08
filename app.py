"""
Smart Education — Streamlit GUI Entry Point
=============================================
Professional interactive dashboard for AI-Based Personalized Learning
Recommendation System. Wraps the existing MultiModelInferenceEngine
for real-time inference with trained PyTorch checkpoints.

Launch: streamlit run app.py
"""

import sys
import os

# Fix Windows encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page Configuration (must be first Streamlit call) ──
st.set_page_config(
    page_title="Smart Education",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal professional CSS ──
st.markdown("""
<style>
    /* ── Global resets ── */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        width: 260px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.15rem;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.35rem 0.5rem;
        border-radius: 6px;
        font-size: 0.88rem;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: rgba(0,0,0,0.04);
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
    }
    .sidebar-subtitle {
        font-size: 0.78rem;
        color: #888;
        margin-bottom: 0.6rem;
    }

    /* ── Metric cards — keep them flat ── */
    div[data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 0;
    }

    /* ── Page header helpers ── */
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.92rem;
        color: #777;
        margin-bottom: 1.2rem;
    }

    /* ── Remove excessive top padding from Streamlit widgets ── */
    div[data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0;
    }

    /* ── Cleaner tab styling ── */
    button[data-baseweb="tab"] {
        font-size: 0.88rem;
    }

    /* ── Alert / status boxes — softer ── */
    .stAlert { border-radius: 8px; }

    /* ── Status dot helpers ── */
    .status-ok { color: #16a34a; font-weight: 600; }
    .status-warn { color: #ca8a04; font-weight: 600; }
    .status-err { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Page Registry ──
PAGES = {
    "Dashboard":              "🏠",
    "Student Analysis":       "🎯",
    "Knowledge State":        "🧠",
    "Recommendations":        "📚",
    "Attention Analysis":     "🎥",
    "Model & Algorithm Lab":  "🔬",
    "Evaluation & Performance": "📊",
    "Data Augmentation":      "⚗️",
    "Dataset Explorer":       "🗃️",
    "System Information":     "⚙️",
}

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎓 Smart Education</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">AI-Based Personalized Learning</div>', unsafe_allow_html=True)

    selected_page = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        format_func=lambda x: f"{PAGES[x]}  {x}",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model loading status — compact
    try:
        from services.inference_service import get_inference_engine
        engine = get_inference_engine()
        import torch
        device_label = "GPU" if torch.cuda.is_available() else "CPU"
        st.caption(f"✅ 5 models + Attention ({device_label})")
        # Ollama Generative AI status indicator
        try:
            from ollama_ai.ollama_client import OllamaClient
            from ollama_ai.ollama_config import OLLAMA_MODEL
            ai_cli = OllamaClient()
            if ai_cli.is_available():
                st.caption(f"🤖 Ollama Active (`{OLLAMA_MODEL}`)")
            else:
                st.caption("🤖 Ollama ")
        except Exception:
            pass
    except Exception as e:
        engine = None
        st.caption(f"❌ Load error: {str(e)[:60]}")

    st.caption("v1.0  ·  `streamlit run app.py`")


# ── Page Router ──
page_modules = {
    "Dashboard":              "gui.page_dashboard",
    "Student Analysis":       "gui.page_student",
    "Knowledge State":        "gui.page_knowledge",
    "Recommendations":        "gui.page_recommendations",
    "Attention Analysis":     "gui.page_attention",
    "Model & Algorithm Lab":  "gui.page_models",
    "Evaluation & Performance": "gui.page_evaluation",
    "Data Augmentation":      "gui.page_augmentation",
    "Dataset Explorer":       "gui.page_dataset",
    "System Information":     "gui.page_system",
}

import importlib
mod = importlib.import_module(page_modules[selected_page])
mod.render(engine)
