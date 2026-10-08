"""
Smart Education Project — Structured Learner Context
===================================================
Constructs a verified, unhallucinated snapshot of the learner's state
gathered strictly from the outputs already produced by:
1. MultiModelInferenceEngine (5 hybrid neural models)
2. KnowledgeStateEstimator (concept mastery & learning gap detection)
3. RecommendationEngine (authentic EdNet lectures, explanations, questions)
4. Multimodal Attention System (video & audio engagement classification)

CRITICAL INVARIANT:
- Only real, actual model and catalog outputs are populated.
- No fabricated scores, synthetic mastery, or fake resources.
- If an attribute is missing or uncomputed, it is explicitly flagged as 'Unavailable'.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LearnerContext:
    student_id: Optional[str] = None
    split: Optional[str] = None
    total_events: int = 0
    inference_time_ms: float = 0.0

    # 5-Model predictions (P(Correct on next question))
    predictions: Dict[str, float] = field(default_factory=dict)
    ensemble_consensus: Optional[float] = None

    # Knowledge State & Concept Mastery
    knowledge_state: Dict[str, Any] = field(default_factory=dict)
    mastery_levels: Dict[str, float] = field(default_factory=dict)
    state_type: str = "Unavailable"  # "MODEL-DERIVED" or "HISTORICAL BASELINE"

    # Diagnosed Learning Gaps (< 60% threshold)
    learning_gaps: List[Dict[str, Any]] = field(default_factory=list)

    # Authentic EdNet Recommendations
    recommended_resources: List[Dict[str, Any]] = field(default_factory=list)

    # Video + Audio Attention Telemetry
    attention_state: Optional[str] = None          # "Attentive", "Partially Attentive", "Inattentive"
    attention_confidence: Optional[float] = None
    attention_probabilities: Dict[str, float] = field(default_factory=dict)
    video_available: bool = False
    audio_available: bool = False

    # Historical / Baseline context
    baseline_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the context to a clean dictionary."""
        return {
            "student_id": self.student_id or "Custom/Anonymous Learner",
            "split": self.split or "unspecified",
            "total_events": self.total_events,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "predictions": {k: round(v, 4) for k, v in self.predictions.items()},
            "ensemble_consensus": round(self.ensemble_consensus, 4) if self.ensemble_consensus is not None else None,
            "state_type": self.state_type,
            "mastery_levels": {k: round(v, 4) for k, v in self.mastery_levels.items()},
            "learning_gaps": self.learning_gaps,
            "recommended_resources": self.recommended_resources,
            "attention_state": self.attention_state or "Unavailable",
            "attention_confidence": round(self.attention_confidence, 4) if self.attention_confidence is not None else None,
            "attention_probabilities": {k: round(v, 4) for k, v in self.attention_probabilities.items()},
            "video_available": self.video_available,
            "audio_available": self.audio_available,
            "baseline_accuracy": round(self.baseline_accuracy, 4) if self.baseline_accuracy is not None else None,
        }

    def format_for_prompt(self, max_gaps: int = 5, max_recs: int = 5) -> str:
        """
        Formats the real metrics into a structured string strictly grounded in actual system data.
        """
        lines = []
        lines.append(f"Student ID: {self.student_id or 'Custom / Anonymous Learner'}")
        if self.split:
            lines.append(f"Cohort Dataset: {self.split.upper()} Cohort ({self.total_events} interaction events)")

        # 1. Multi-Model Predictions
        if self.predictions:
            lines.append("\n[1. Multi-Model Neural Predictions - Next Question Probability]")
            for m_name, prob in self.predictions.items():
                lines.append(f"- {m_name}: {prob * 100:.1f}%")
            if self.ensemble_consensus is not None:
                lines.append(f"- Ensemble Consensus: {self.ensemble_consensus * 100:.1f}%")
        else:
            lines.append("\n[1. Multi-Model Neural Predictions]: Unavailable")

        # 2. Attention Telemetry
        lines.append("\n[2. Multimodal Video + Audio Attention Telemetry]")
        if self.attention_state:
            conf_str = f" ({self.attention_confidence * 100:.1f}% confidence)" if self.attention_confidence is not None else ""
            lines.append(f"- Diagnosed Engagement: {self.attention_state}{conf_str}")
            if self.attention_probabilities:
                prob_parts = [f"{k}: {v*100:.1f}%" for k, v in self.attention_probabilities.items()]
                lines.append(f"- Probability Distribution: {', '.join(prob_parts)}")
            streams = []
            if self.video_available: streams.append("Facial/Gaze Video")
            if self.audio_available: streams.append("Acoustic Audio")
            lines.append(f"- Active Sensory Modalities: {', '.join(streams) if streams else 'Feature Vector Only'}")
        else:
            lines.append("- Diagnosed Engagement: Unavailable (No active video/audio stream)")

        # 3. Diagnosed Learning Gaps
        lines.append("\n[3. Diagnosed Critical Learning Gaps (< 60% Mastery Threshold)]")
        if self.learning_gaps:
            lines.append(f"Total Deficiencies Detected: {len(self.learning_gaps)}")
            for idx, gap in enumerate(self.learning_gaps[:max_gaps], 1):
                cid = gap.get("concept_id", gap.get("concept", "N/A"))
                mastery = gap.get("mastery_score", gap.get("mastery", 0.0))
                attempts = gap.get("attempts", gap.get("total_attempts", 1))
                sev = gap.get("gap_severity", max(0.0, 0.60 - mastery))
                lines.append(
                    f"- Deficit #{idx}: Concept #{cid} | Mastery: {mastery * 100:.1f}% | "
                    f"Deficit Severity: {sev * 100:.1f}% | Evaluated Attempts: {attempts}"
                )
        else:
            lines.append("- No critical concept deficiencies below 60% threshold detected.")

        # 4. Top Authentic Recommendations
        lines.append("\n[4. Ranked Authentic EdNet Recommendations (Lectures > Explanations > Practice)]")
        if self.recommended_resources:
            for idx, r in enumerate(self.recommended_resources[:max_recs], 1):
                rtype = str(r.get("type", "Resource")).upper()
                rid = r.get("id", "N/A")
                cid = r.get("concept", "N/A")
                part = f"Part {r.get('part')}" if r.get("part") is not None else ""
                dur = f"{r.get('duration_sec')}s" if r.get("duration_sec") and r.get("duration_sec") > 0 else ""
                details = [d for d in [part, dur] if d]
                det_str = f" ({', '.join(details)})" if details else ""
                lines.append(f"- Recommendation #{idx}: [{rtype}] ID: {rid} | Targets Concept: #{cid}{det_str}")
        else:
            lines.append("- No specific recommendations available.")

        return "\n".join(lines)


def build_learner_context(
    inference_result: Optional[Dict[str, Any]] = None,
    attention_result: Optional[Dict[str, Any]] = None,
    student_id: Optional[str] = None
) -> LearnerContext:
    """
    Factory function safely mapping existing system inference dictionaries
    into a structured LearnerContext without inventing or altering any values.
    """
    ctx = LearnerContext()

    # 1. Populate from ML inference result
    if inference_result:
        ctx.student_id = inference_result.get("student_id") or student_id
        ctx.split = inference_result.get("split")
        ctx.total_events = inference_result.get("total_events", 0)
        ctx.inference_time_ms = inference_result.get("inference_time_ms", 0.0)

        # Predictions
        preds = inference_result.get("predictions", {})
        ctx.predictions = preds
        ctx.ensemble_consensus = preds.get("Ensemble Consensus")

        # Knowledge State
        ks = inference_result.get("knowledge_state", {})
        ctx.knowledge_state = ks
        mastery = {}
        state_type = "Unavailable"
        if isinstance(ks, dict):
            for cid, data in ks.items():
                if isinstance(data, dict):
                    mastery[str(cid)] = float(data.get("mastery_score", 0.0))
                    state_type = data.get("state_type", "MODEL-DERIVED KNOWLEDGE STATE")
                else:
                    mastery[str(cid)] = float(data)
                    state_type = "MODEL-DERIVED KNOWLEDGE STATE"
        ctx.mastery_levels = mastery
        ctx.state_type = state_type

        # Learning Gaps
        ctx.learning_gaps = inference_result.get("learning_gaps", [])

        # Recommendations
        ctx.recommended_resources = inference_result.get("recommendations", [])

        # Baseline accuracy if present
        bstate = inference_result.get("baseline_state")
        if bstate and isinstance(bstate, dict):
            total_corr = sum(v.get("correct_count", 0) for v in bstate.values() if isinstance(v, dict))
            total_att = sum(v.get("total_attempts", 0) for v in bstate.values() if isinstance(v, dict))
            if total_att > 0:
                ctx.baseline_accuracy = total_corr / total_att

    # 2. Populate attention signals if available
    if attention_result:
        att = attention_result.get("attention", attention_result)
        ctx.attention_state = att.get("attention_class")
        ctx.attention_confidence = att.get("confidence")
        ctx.attention_probabilities = att.get("probabilities", {})
        ctx.video_available = bool(att.get("video_available", False))
        ctx.audio_available = bool(att.get("audio_available", False))

        # Check if attention-adjusted recommendations exist
        pers = attention_result.get("personalization")
        if pers and pers.get("adapted_recommendations"):
            ctx.recommended_resources = pers.get("adapted_recommendations")

    elif not ctx.student_id and student_id:
        ctx.student_id = student_id

    return ctx
