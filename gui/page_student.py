import streamlit as st
import pandas as pd
import plotly.express as px
from services.inference_service import predict_student, predict_custom

def render_prediction_results(result):
    if not result:
        st.warning("No prediction results available.")
        return

    # a) Prediction comparison: Plotly horizontal bar chart showing all 5 model probabilities + ensemble
    st.subheader("Model Predictions")
    preds = result.get('predictions', {})
    if preds:
        df_preds = pd.DataFrame({
            "Model": list(preds.keys()),
            "Probability": list(preds.values())
        })
        
        # Sort by probability
        df_preds = df_preds.sort_values("Probability")
        
        fig = px.bar(
            df_preds, 
            x="Probability", 
            y="Model", 
            orientation="h",
            title="Correctness Probability by Model",
            color="Probability",
            color_continuous_scale="Viridis",
            range_x=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions found in result.")

    # b) Inference latency metric
    latency = result.get('inference_time_ms', 0)
    st.metric(label="Inference Latency", value=f"{latency:.2f} ms")

    # c) Knowledge state: show top concepts as a dataframe with concept_id, mastery_score
    st.subheader("Knowledge State")
    k_state = result.get('knowledge_state', {})
    if k_state:
        # Convert dictionary format to dataframe
        if isinstance(k_state, dict):
            # If it's a dict of dicts where outer key is concept_id
            first_val = next(iter(k_state.values()))
            if isinstance(first_val, dict):
                df_k = pd.DataFrame.from_dict(k_state, orient='index').reset_index()
                if 'concept_id' not in df_k.columns and 'index' in df_k.columns:
                    df_k = df_k.rename(columns={'index': 'concept_id'})
            else:
                # Flat dict of concept_id -> score
                df_k = pd.DataFrame([{"concept_id": k, "mastery_score": v} for k, v in k_state.items()])
        else:
            df_k = pd.DataFrame(k_state)
            
        if 'mastery_score' in df_k.columns:
            df_k = df_k.sort_values(by='mastery_score', ascending=False)
            st.dataframe(df_k[['concept_id', 'mastery_score']].head(20), use_container_width=True)
        else:
            st.dataframe(df_k.head(20), use_container_width=True)
    else:
        st.info("No knowledge state data available.")

    # d) Learning gaps: warning-styled table with gap severity
    st.subheader("Learning Gaps")
    gaps = result.get('learning_gaps', [])
    if gaps:
        df_gaps = pd.DataFrame(gaps)
        st.warning(f"Identified {len(gaps)} potential learning gaps.")
        st.dataframe(df_gaps, use_container_width=True)
    else:
        st.success("No significant learning gaps identified.")

    # e) Recommendations: table with type, resource ID, concept, part
    st.subheader("Recommendations")
    recs = result.get('recommendations', [])
    if recs:
        df_recs = pd.DataFrame(recs)
        st.dataframe(df_recs, use_container_width=True)
    else:
        st.info("No specific recommendations at this time.")

    # f) AI Tutor & Educational Dialog
    st.markdown("---")
    with st.expander("🤖 AI Tutor — Ask Questions About Your Learning Profile", expanded=False):
        st.caption("Ask questions about your concept mastery, recommendations, or study routine. All answers are strictly grounded in your actual ML metrics.")
        
        # Quick prompt buttons
        q_cols = st.columns(3)
        quick_query = None
        with q_cols[0]:
            if st.button("❓ Why am I weak in my top gap?", key="quick_q1"):
                quick_query = "Why am I struggling with my weakest concept, and what should I review?"
        with q_cols[1]:
            if st.button("📚 Explain my recommendations", key="quick_q2"):
                quick_query = "Why did the system recommend these specific video lectures and questions?"
        with q_cols[2]:
            if st.button("📅 Give me a study plan today", key="quick_q3"):
                quick_query = "Please generate a personalized study plan for today targeting my weak concepts."

        user_input = st.text_input("Your question for AI Tutor:", value=quick_query or "", placeholder="e.g., How can I improve my mastery on Concept #18?", key="tutor_user_query")
        
        if st.button("💬 Ask AI Tutor", key="btn_ask_tutor") and user_input:
            with st.spinner("AI Tutor is analyzing your diagnostic profile..."):
                from ollama_ai.learner_context import build_learner_context
                from ollama_ai.tutor import AITutor
                ai_ctx = build_learner_context(inference_result=result, student_id=result.get("student_id"))
                tutor = AITutor()
                tutor_resp = tutor.ask(ai_ctx, query=user_input)
                st.session_state["last_tutor_response"] = tutor_resp
                st.session_state["last_tutor_query"] = user_input

        if "last_tutor_response" in st.session_state:
            resp_obj = st.session_state["last_tutor_response"]
            badge = "🟢 Ollama AI Tutor" if resp_obj.get("ai_available") else "⚪ Deterministic Pedagogical Tutor (Ollama Offline)"
            st.markdown(f"**Query:** *\"{st.session_state.get('last_tutor_query', '')}\"*")
            st.markdown(f"**Source:** `{badge}`")
            st.markdown(resp_obj.get("response", ""))



def render(engine):
    st.title("Student Analysis")
    
    if engine is None:
        st.error("Inference engine could not be loaded. Please check the backend.")
        return

    tab1, tab2 = st.tabs(["Student ID Lookup", "Custom Input"])
    
    with tab1:
        st.header("Student ID Lookup")
        
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID", placeholder="e.g., u1165")
        with col2:
            split = st.selectbox("Split", options=["test", "train", "val"])
            
        if st.button("Run Inference", key="run_inf_lookup"):
            if student_id:
                with st.spinner("Running inference..."):
                    # Note: predict_student currently might not take 'split' depending on the interface,
                    # but if it does, we pass it, otherwise assume it uses the best available data.
                    # As per instructions, it's called as `predict_student(engine, student_id)`
                    result = predict_student(engine, student_id)
                    if result:
                        st.session_state['last_prediction'] = result
                        render_prediction_results(result)
                    else:
                        st.error(f"Could not find data for student: {student_id} or inference failed.")
            else:
                st.warning("Please enter a Student ID.")
                
    with tab2:
        st.header("Custom Input Inference")
        
        num_events = st.number_input("Number of events", min_value=1, max_value=20, value=5, step=1)
        
        st.write("Configure recent events:")
        
        events = []
        for i in range(int(num_events)):
            with st.expander(f"Event {i+1}", expanded=(i<2)):
                col_q, col_c, col_cor, col_rt = st.columns(4)
                
                with col_q:
                    q_idx = st.number_input("Question Index", min_value=0, max_value=11334, value=i*10, key=f"q_{i}")
                with col_c:
                    c_tag = st.number_input("Concept Tag", min_value=0, max_value=188, value=18, key=f"c_{i}")
                with col_cor:
                    is_corr = st.checkbox("Is Correct", value=True, key=f"cor_{i}")
                with col_rt:
                    rt_sec = st.number_input("Response Time (sec)", min_value=0.0, value=15.0, step=1.0, key=f"rt_{i}")
                    
                # Build proper normalization: response_time = (rt_val - 20.0) / 15.0
                norm_rt = (rt_sec - 20.0) / 15.0
                
                events.append({
                    "question_id": q_idx,
                    "part": 1,
                    "tags": [c_tag],
                    "is_correct": int(is_corr),
                    "response_time": norm_rt,
                    "source_encoded": 1,
                    "platform_encoded": 1
                })
                
        if st.button("Predict", key="run_inf_custom"):
            with st.spinner("Running inference on custom events..."):
                result = predict_custom(engine, events)
                if result:
                    st.session_state['last_prediction'] = result
                    render_prediction_results(result)
                else:
                    st.error("Inference failed for custom input.")
