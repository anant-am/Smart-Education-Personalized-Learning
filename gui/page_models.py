import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from services.inference_service import (
    get_model_results,
    get_all_model_results,
    get_plot_path,
    get_experiment_manifest
)

def render(engine):
    st.title("Model & Algorithm Lab")
    
    st.header("Section A: Model Cards")
    
    model_results = get_all_model_results()
    if not model_results:
        st.warning("No model results available.")
    else:
        # Create tabs for models 1 to 5
        tab_names = [f"Model {i}" for i in range(1, 6)]
        tabs = st.tabs(tab_names)
        
        for i, tab in enumerate(tabs, start=1):
            with tab:
                result = get_model_results(i)
                if not result:
                    st.warning(f"Results for Model {i} not found.")
                    continue
                
                st.subheader(result.get('model_name', f"Model {i}"))
                st.write(f"**Parameters:** {result.get('parameters', 'Unknown')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Test Metrics")
                    metrics = result.get('test_metrics', {})
                    if not metrics:
                        # Fallback for model 4 or if test_metrics is missing
                        manifest = get_experiment_manifest()
                        if manifest and str(i) in manifest.get('models', {}):
                            metrics = manifest['models'][str(i)].get('metrics', {})
                            
                    if not metrics:
                        st.info("No test metrics available.")
                    else:
                        for k, v in metrics.items():
                            st.metric(k.replace('_', ' ').title(), f"{v:.4f}" if isinstance(v, (int, float)) else v)
                
                with col2:
                    st.markdown("### Diagnosis & Correction")
                    diagnosis = result.get('diagnosis', {})
                    if diagnosis:
                        st.json(diagnosis)
                    else:
                        st.info("No diagnosis available.")
                        
                    correction = result.get('correction_info', {})
                    if correction:
                        st.json(correction)
                    else:
                        st.info("No correction info available.")
                
                st.markdown("### Training Curves")
                curve_col1, curve_col2 = st.columns(2)
                with curve_col1:
                    initial_plot = get_plot_path(f'model{i}_initial_curves.png')
                    if initial_plot and initial_plot.exists():
                        st.image(str(initial_plot), caption="Initial Curves")
                    else:
                        st.info("Initial curves plot not found.")
                with curve_col2:
                    corrected_plot = get_plot_path(f'model{i}_corrected_curves.png')
                    if corrected_plot and corrected_plot.exists():
                        st.image(str(corrected_plot), caption="Corrected Curves")
                    else:
                        st.info("Corrected curves plot not found.")
                        
                st.markdown("### Training History (Interactive)")
                history_col1, history_col2 = st.columns(2)
                with history_col1:
                    initial_history = result.get('initial_history', {})
                    if initial_history:
                        fig = go.Figure()
                        if 'train_loss' in initial_history:
                            fig.add_trace(go.Scatter(y=initial_history['train_loss'], name='Train Loss'))
                        if 'val_loss' in initial_history:
                            fig.add_trace(go.Scatter(y=initial_history['val_loss'], name='Val Loss'))
                        if 'train_auc' in initial_history:
                            fig.add_trace(go.Scatter(y=initial_history['train_auc'], name='Train AUC'))
                        if 'val_auc' in initial_history:
                            fig.add_trace(go.Scatter(y=initial_history['val_auc'], name='Val AUC'))
                        fig.update_layout(title="Initial Training History")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No initial history data.")
                
                with history_col2:
                    retrained_history = result.get('retrained_history', {})
                    if retrained_history:
                        fig = go.Figure()
                        if 'train_loss' in retrained_history:
                            fig.add_trace(go.Scatter(y=retrained_history['train_loss'], name='Train Loss'))
                        if 'val_loss' in retrained_history:
                            fig.add_trace(go.Scatter(y=retrained_history['val_loss'], name='Val Loss'))
                        if 'train_auc' in retrained_history:
                            fig.add_trace(go.Scatter(y=retrained_history['train_auc'], name='Train AUC'))
                        if 'val_auc' in retrained_history:
                            fig.add_trace(go.Scatter(y=retrained_history['val_auc'], name='Val AUC'))
                        fig.update_layout(title="Retrained History")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No retrained history data.")
                        
                st.markdown("### Cross-Validation Summary")
                cv_summary = result.get('cv_summary', {})
                if cv_summary:
                    cv_df = pd.DataFrame([cv_summary])
                    st.dataframe(cv_df, use_container_width=True)
                else:
                    st.info("No CV summary available.")

    st.header("Section B: Algorithm Implementation Status")
    with st.expander("View Algorithm Status", expanded=True):
        algo_data = [
            {"Algorithm": "SMOTE", "Status": "IMPLEMENTED", "Source File": "src/smote_utils.py", "Description": "Synthetic Minority Oversampling"},
            {"Algorithm": "ADASYN", "Status": "IMPLEMENTED", "Source File": "src/smote_utils.py", "Description": "Adaptive Density-Based Sampling"},
            {"Algorithm": "WGAN-GP", "Status": "IMPLEMENTED", "Source File": "src/gan_augmentation.py", "Description": "Tabular Conditional GAN"},
            {"Algorithm": "10-Fold Grouped CV", "Status": "IMPLEMENTED", "Source File": "src/cross_validation.py", "Description": "GroupKFold by user_id"},
            {"Algorithm": "Leakage Checks", "Status": "IMPLEMENTED", "Source File": "src/leakage_checks.py", "Description": "6-type leakage auditor"},
            {"Algorithm": "Overfitting Detection", "Status": "IMPLEMENTED", "Source File": "src/trainer.py", "Description": "Configurable threshold diagnosis"},
            {"Algorithm": "Knowledge State (Dual)", "Status": "IMPLEMENTED", "Source File": "src/knowledge_state.py", "Description": "Model-derived + Historical baseline"},
            {"Algorithm": "Recommendation Engine", "Status": "IMPLEMENTED", "Source File": "src/recommendation.py", "Description": "Concept dependency graph + ranking"}
        ]
        
        algo_df = pd.DataFrame(algo_data)
        
        def color_status(val):
            color = 'green' if val == 'IMPLEMENTED' else 'black'
            return f'color: {color}'
        
        # Use a try-except to handle pandas version differences for styling
        try:
            if hasattr(algo_df.style, 'map'):
                styled_df = algo_df.style.map(color_status, subset=['Status'])
            else:
                styled_df = algo_df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
        except Exception:
            st.dataframe(algo_df, use_container_width=True)
