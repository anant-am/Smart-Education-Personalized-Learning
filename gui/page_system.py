"""
Smart Education - System Information & Project Configuration Page
================================================================
Displays host hardware/software runtime diagnostics, filesystem
paths, trained model checkpoint registries, university academic
compliance audits, future work mappings, and publication-ready plots.
"""

import sys
import time
import platform
from pathlib import Path
from typing import Optional, Any, Dict, List

import pandas as pd
import streamlit as st
import plotly.express as px

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
)
from services.inference_service import (
    get_system_info,
    get_checkpoint_files,
    get_report_markdown,
    get_available_plots,
    get_plot_path,
)


def render(engine: Optional[Any] = None) -> None:
    """
    Renders the System Information & Project Configuration GUI page.

    Parameters:
        engine: MultiModelInferenceEngine instance or None if loading failed.
    """
    # 1. Page Header
    st.title("System Information")
    st.caption("Hardware diagnostics, GPU status, checkpoint files, compliance audits, and plot catalog.")

    # 2. Hardware & Software
    st.subheader("Hardware & Software")

    if engine is None:
        st.warning(
            "Inference engine is not initialized (offline mode). "
            "Displaying host environment without GPU acceleration details."
        )
        sys_info: Dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "pytorch_version": "N/A",
            "cuda_version": "N/A",
            "streamlit_version": getattr(st, "__version__", "N/A"),
            "gpu_name": "N/A (Engine offline)",
            "gpu_vram": "N/A (Engine offline)",
            "device": "Offline (CPU)",
            "os": platform.platform(),
        }
        try:
            import torch
            sys_info["pytorch_version"] = torch.__version__
        except Exception:
            pass
    else:
        try:
            raw_info = get_system_info(engine)
        except Exception:
            raw_info = {}

        python_ver = raw_info.get("python_version", sys.version.split()[0])
        pytorch_ver = raw_info.get("pytorch_version", "N/A")
        cuda_ver = (
            raw_info.get("cuda_version", "N/A")
            if raw_info.get("cuda_available")
            else "None (CPU Mode)"
        )
        streamlit_ver = raw_info.get("streamlit_version", getattr(st, "__version__", "N/A"))

        gpu_name = raw_info.get("gpu_name", "None Detected (CPU Mode)")
        if "gpu_vram_gb" in raw_info:
            gpu_vram = f"{raw_info['gpu_vram_gb']:.2f} GB"
        else:
            gpu_vram = "N/A (CPU Mode)"

        device_val = raw_info.get("device", "cpu")
        os_val = raw_info.get("os", platform.platform())

        sys_info = {
            "python_version": python_ver,
            "pytorch_version": pytorch_ver,
            "cuda_version": cuda_ver,
            "streamlit_version": streamlit_ver,
            "gpu_name": gpu_name,
            "gpu_vram": gpu_vram,
            "device": device_val,
            "os": os_val,
        }

    # 2-column layout with st.metric
    col_sw, col_hw = st.columns(2)

    with col_sw:
        st.markdown("##### Software")
        st.metric("Python version", sys_info["python_version"])
        st.metric("PyTorch version", sys_info["pytorch_version"])
        st.metric("CUDA version", sys_info["cuda_version"])
        st.metric("Streamlit version", sys_info["streamlit_version"])

    with col_hw:
        st.markdown("##### Hardware")
        st.metric("GPU name", sys_info["gpu_name"])
        st.metric("VRAM", sys_info["gpu_vram"])
        st.metric("Device", sys_info["device"])
        st.metric("OS", sys_info["os"])

    # 3. Project Paths
    st.divider()
    st.subheader("Project Paths")

    paths_data = [
        {
            "Directory": "Project root",
            "Path": str(PROJECT_ROOT),
            "Status": "Exists" if PROJECT_ROOT.exists() else "Missing",
            "Items": sum(1 for _ in PROJECT_ROOT.iterdir()) if PROJECT_ROOT.exists() else 0,
        },
        {
            "Directory": "Data dir",
            "Path": str(DATA_DIR),
            "Status": "Exists" if DATA_DIR.exists() else "Missing",
            "Items": sum(1 for _ in DATA_DIR.iterdir()) if DATA_DIR.exists() else 0,
        },
        {
            "Directory": "Models dir",
            "Path": str(MODELS_DIR),
            "Status": "Exists" if MODELS_DIR.exists() else "Missing",
            "Items": sum(1 for _ in MODELS_DIR.iterdir()) if MODELS_DIR.exists() else 0,
        },
        {
            "Directory": "Reports dir",
            "Path": str(REPORTS_DIR),
            "Status": "Exists" if REPORTS_DIR.exists() else "Missing",
            "Items": sum(1 for _ in REPORTS_DIR.iterdir()) if REPORTS_DIR.exists() else 0,
        },
        {
            "Directory": "Plots dir",
            "Path": str(PLOTS_DIR),
            "Status": "Exists" if PLOTS_DIR.exists() else "Missing",
            "Items": sum(1 for _ in PLOTS_DIR.iterdir()) if PLOTS_DIR.exists() else 0,
        },
    ]
    df_paths = pd.DataFrame(paths_data)
    st.dataframe(df_paths, use_container_width=True, hide_index=True)

    # 4. Checkpoint Files
    st.divider()
    st.subheader("Checkpoint Files")

    ckpts = get_checkpoint_files()
    if ckpts:
        total_files = len(ckpts)
        total_mb = sum(c.get("size_mb", 0.0) for c in ckpts)

        # Count total: X checkpoint files, Y MB total
        st.info(f"Count total: {total_files} checkpoint files, {total_mb:.2f} MB total")

        df_ckpts = pd.DataFrame([
            {
                "Filename": c["name"],
                "Size (MB)": c["size_mb"],
                "Last Modified": c["modified"],
                "Extension": c.get("extension", ""),
            }
            for c in ckpts
        ])

        # Filter option by extension
        ext_list = ["All"] + sorted(list({c.get("extension", "") for c in ckpts if c.get("extension")}))
        sel_ext = st.selectbox("Filter by extension:", options=ext_list, index=0)
        if sel_ext != "All":
            view_df = df_ckpts[df_ckpts["Extension"] == sel_ext]
        else:
            view_df = df_ckpts

        st.dataframe(view_df, use_container_width=True, hide_index=True)

        # Plotly chart of model checkpoint weights
        pth_files = [c for c in ckpts if c.get("extension") in [".pth", ".pt"]]
        if pth_files:
            pth_df = pd.DataFrame(pth_files).sort_values(by="size_mb", ascending=True)
            fig_ckpts = px.bar(
                pth_df,
                x="size_mb",
                y="name",
                orientation="h",
                title="Model Checkpoints Storage Size (MB)",
                labels={"size_mb": "Size (MB)", "name": "Checkpoint File"},
                color="size_mb",
                color_continuous_scale="Blues",
            )
            fig_ckpts.update_layout(
                height=max(360, len(pth_df) * 22),
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="File Size (MB)",
                yaxis_title="Checkpoint Name",
            )
            st.plotly_chart(fig_ckpts, use_container_width=True)
    else:
        st.warning("No checkpoint files found in models/checkpoints directory.")

    # 5. Compliance Checklist
    st.divider()
    st.subheader("Compliance Checklist")

    with st.expander("Academic Compliance Checklist (final_compliance_checklist.md)", expanded=False):
        checklist_md = get_report_markdown("final_compliance_checklist.md")
        if checklist_md and checklist_md.strip():
            st.markdown(checklist_md)
        else:
            st.warning("Compliance checklist report ('final_compliance_checklist.md') not found.")

    # 6. Future Work Mapping
    st.subheader("Future Work Mapping")

    with st.expander("Future Work Literature Mapping (future_work_mapping.md)", expanded=False):
        future_work_md = get_report_markdown("future_work_mapping.md")
        if future_work_md and future_work_md.strip():
            st.markdown(future_work_md)
        else:
            st.warning("Future work mapping report ('future_work_mapping.md') not found.")

    # 7. Available Plots
    st.divider()
    st.subheader("Available Plots")

    available_plots = get_available_plots()
    if available_plots:
        st.caption(f"Catalog of {len(available_plots)} publication-ready figures available in plots/ directory.")
        selected_plot = st.selectbox(
            "Select plot to view:",
            options=available_plots,
            index=0,
            help="Choose a plot to display.",
        )
        if selected_plot:
            plot_path = get_plot_path(selected_plot)
            if plot_path and plot_path.exists():
                st.image(
                    str(plot_path),
                    caption=f"Plot: {selected_plot}",
                    use_container_width=True,
                )
                try:
                    stat = plot_path.stat()
                    size_kb = stat.st_size / 1024
                    mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                    st.caption(f"File: {selected_plot} | Size: {size_kb:.1f} KB | Last Modified: {mod_time}")
                except Exception:
                    pass
            else:
                st.warning(f"Plot file '{selected_plot}' was not found at {plot_path}.")
    else:
        st.warning("No plot files found in plots directory.")
