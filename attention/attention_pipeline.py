"""
Smart Education Project — Video + Audio Attention & Personalization Pipeline
=============================================================================
High-level integration bridge uniting:
1. Video + Audio Attention Detection
2. EdNet Multi-Model Knowledge State Estimation
3. Concept Learning Gap Detection
4. Attention-Aware Pedagogical Recommendation Adjustment

Main function:
  run_video_attention_pipeline(video_path, student_id=..., ...)
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_inference import AttentionInferenceEngine


def adjust_recommendations_by_attention(
    original_recommendations: List[Dict[str, Any]],
    attention_class: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Applies pedagogical modulation to authentic EdNet recommendations
    based on real-time learner attention without replacing or corrupting
    the underlying recommender engine.

    Grounded in Cognitive Load Theory & Engagement-Paced Remediation:
    - Inattentive: Prioritize bite-sized video lectures (duration <= 300s) or explanations.
    - Partially Attentive: Balance explanations and guided practice.
    - Attentive: Prioritize active practice questions and comprehensive lectures.
    """
    adapted = []

    if attention_class == "Inattentive":
        strategy_title = "Cognitive Relief & Foundations (Bite-Sized Lectures & Explanations)"
        rationale = (
            f"Learner is currently Inattentive (Confidence: {confidence*100:.1f}%). "
            "To prevent cognitive overload and re-engage attention, foundational video lectures "
            "under 5 minutes and concise worked explanations are prioritized over lengthy question bundles."
        )
        priority_order = {"lecture": 1, "explanation": 2, "question": 3}

        # Sort and filter for shorter durations where possible
        def inattentive_key(r):
            p = priority_order.get(r.get("type", "resource"), 9)
            dur = r.get("duration_sec", 999)
            return (p, dur if dur > 0 else 999)

        adapted = sorted(original_recommendations, key=inattentive_key)

        # Mark recommendation pacing
        for r in adapted:
            r_copy = dict(r)
            if r_copy.get("type") == "lecture" and r_copy.get("duration_sec", 0) <= 300:
                r_copy["pacing_tag"] = "Bite-Sized Video"
            elif r_copy.get("type") == "explanation":
                r_copy["pacing_tag"] = "Direct Concept Explanation"
            else:
                r_copy["pacing_tag"] = "Standard Practice"

    elif attention_class == "Partially Attentive":
        strategy_title = "Guided Pacing (Balanced Worked Solutions & Targeted Practice)"
        rationale = (
            f"Learner is Partially Attentive (Confidence: {confidence*100:.1f}%). "
            "Providing a structured mix of worked explanations and single practice questions "
            "to reinforce concept mastery."
        )
        priority_order = {"explanation": 1, "question": 2, "lecture": 3}
        adapted = sorted(original_recommendations, key=lambda r: priority_order.get(r.get("type", "resource"), 9))

    else:  # Attentive
        strategy_title = "Accelerated Mastery (Active Practice & Deep-Dive Exploration)"
        rationale = (
            f"Learner is Attentive (Confidence: {confidence*100:.1f}%). "
            "Engagement is optimal; presenting active practice questions to test retrieval "
            "and deep-dive lectures to consolidate mastery."
        )
        priority_order = {"question": 1, "lecture": 2, "explanation": 3}
        adapted = sorted(original_recommendations, key=lambda r: priority_order.get(r.get("type", "resource"), 9))

    return {
        "adapted_recommendations": adapted,
        "strategy_title": strategy_title,
        "rationale": rationale,
        "attention_state": attention_class,
        "confidence": confidence
    }


def run_video_attention_pipeline(
    video_path: Optional[str] = None,
    student_id: Optional[str] = None,
    sequence_events: Optional[List[Dict[str, Any]]] = None,
    attention_engine: Optional[AttentionInferenceEngine] = None,
    kt_engine: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Unified master entry point integrating:
      Video File -> Multimodal Attention -> EdNet Knowledge State -> Adjusted Recommendations
    """
    t_start = time.time()
    result = {}

    # 1. Attention Inference
    if attention_engine is None:
        try:
            attention_engine = AttentionInferenceEngine()
        except Exception as e:
            attention_engine = None
            result["attention_error"] = str(e)

    if attention_engine is not None and video_path is not None:
        att_res = attention_engine.predict_video(video_path)
    else:
        att_res = {
            "status": "No video provided",
            "attention_class": "Attentive",
            "confidence": 1.0,
            "probabilities": {"Inattentive": 0.0, "Partially Attentive": 0.0, "Attentive": 1.0},
            "video_available": False,
            "audio_available": False,
            "warnings": ["Video path not specified; using neutral attention context"]
        }

    result["attention"] = att_res

    # 2. Knowledge State & Learning Gap Inference
    kt_result = None
    if student_id or sequence_events:
        if kt_engine is None:
            try:
                from main import MultiModelInferenceEngine
                kt_engine = MultiModelInferenceEngine()
            except Exception as e:
                kt_engine = None
                result["kt_error"] = str(e)

        if kt_engine is not None:
            if student_id:
                kt_result = kt_engine.predict_student_by_id(student_id)
            elif sequence_events:
                kt_result = kt_engine.predict_sequence(sequence_events)

    if kt_result:
        result["student_id"] = student_id or "Custom_Simulated_Learner"
        result["predictions"] = kt_result.get("predictions", {})
        result["knowledge_state"] = kt_result.get("knowledge_state", {})
        result["learning_gaps"] = kt_result.get("learning_gaps", [])
        result["original_recommendations"] = kt_result.get("recommendations", [])
        result["total_events"] = kt_result.get("total_events", 0)

        # 3. Attention-Aware Personalization Modulation
        att_class = att_res.get("attention_class", "Attentive")
        att_conf = att_res.get("confidence", 0.5)

        modulation = adjust_recommendations_by_attention(
            original_recommendations=kt_result.get("recommendations", []),
            attention_class=att_class,
            confidence=att_conf
        )
        result["personalization"] = modulation
    else:
        result["student_id"] = None
        result["knowledge_state"] = {}
        result["learning_gaps"] = []
        result["original_recommendations"] = []
        result["personalization"] = None

    result["pipeline_execution_time_ms"] = (time.time() - t_start) * 1000
    return result
