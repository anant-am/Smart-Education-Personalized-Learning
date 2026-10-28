# -*- coding: utf-8 -*-
"""
Smart Education — GUI Services Package
======================================
Exposes inference and experiment services for the Streamlit GUI.
"""

from services.experiment_service import (
    get_experiment_results_df,
    get_five_model_comparison,
    get_final_comparison,
    get_augmentation_comparison,
    get_experiment_manifest,
    get_model_results,
    get_all_model_results,
    get_checkpoint_files,
    get_checkpoint_metadata,
    get_attention_results,
    get_figure_path,
    get_available_figures,
    get_plot_path,
    get_available_plots,
)

__all__ = [
    "get_experiment_results_df",
    "get_five_model_comparison",
    "get_final_comparison",
    "get_augmentation_comparison",
    "get_experiment_manifest",
    "get_model_results",
    "get_all_model_results",
    "get_checkpoint_files",
    "get_checkpoint_metadata",
    "get_attention_results",
    "get_figure_path",
    "get_available_figures",
    "get_plot_path",
    "get_available_plots",
]
