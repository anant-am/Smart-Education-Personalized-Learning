"""
Smart Education Project — Knowledge State Estimation & Learning Gap Detection
=============================================================================
Provides:
1. Model-Derived Knowledge State: Estimates per-concept mastery from deep KT
   model output probabilities across chronological interactions.
2. Historical Baseline Knowledge State: Computes transparent empirical accuracy
   per concept tag from historical attempts.
3. Configurable Learning Gap Detection: Identifies concepts with mastery below
   threshold with a configurable minimum attempt requirement, sorted by deficit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class KnowledgeStateEstimator:
    """
    Estimates student knowledge state and identifies high-priority learning gaps.
    Distinguishes strictly between Model-Derived and Historical Baseline states.
    """
    def __init__(self, tag_to_name: Optional[Dict[int, str]] = None):
        self.tag_to_name = tag_to_name or {}

    def compute_model_derived_state(self, model_predictions: List[float],
                                    question_tags: List[List[int]],
                                    concept_weights: Optional[List[float]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Computes MODEL-DERIVED KNOWLEDGE STATE:
        Aggregates model sigmoid-activated next-response predictions for questions
        tagged with each concept.
        """
        tag_scores = {}
        tag_counts = {}

        for i, (pred, tags) in enumerate(zip(model_predictions, question_tags)):
            w = concept_weights[i] if concept_weights is not None else 1.0
            for tag in tags:
                if tag <= 0:
                    continue
                tag_scores[tag] = tag_scores.get(tag, 0.0) + (pred * w)
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        model_state = {}
        for tag, total in tag_scores.items():
            count = tag_counts[tag]
            mastery = total / count
            model_state[tag] = {
                'concept_id': tag,
                'concept_name': self.tag_to_name.get(tag, f"Concept_{tag}"),
                'mastery_score': float(mastery),
                'predicted_observations': count,
                'state_type': 'MODEL-DERIVED KNOWLEDGE STATE'
            }
        return model_state

    def compute_historical_baseline_state(self, student_events: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Computes HISTORICAL BASELINE KNOWLEDGE STATE:
        Calculates empirical accuracy (correct / attempts) directly from logged events.
        """
        tag_history = {}

        if 'enter_ts' in student_events.columns:
            events = student_events.sort_values('enter_ts')
        elif 'timestamp' in student_events.columns:
            events = student_events.sort_values('timestamp')
        else:
            events = student_events

        for _, row in events.iterrows():
            corr = int(row['is_correct'])
            raw_tags = row.get('tags')
            tag_list = []

            if isinstance(raw_tags, str):
                try:
                    tag_list = [int(t.strip()) for t in raw_tags.split(';') if t.strip().isdigit()]
                except Exception:
                    pass
            elif isinstance(raw_tags, (list, tuple, np.ndarray)):
                tag_list = [int(t) for t in raw_tags if str(t).isdigit()]
            elif pd.notna(raw_tags) and str(raw_tags).isdigit():
                tag_list = [int(raw_tags)]

            for t in tag_list:
                if t <= 0:
                    continue
                if t not in tag_history:
                    tag_history[t] = []
                tag_history[t].append(corr)

        baseline_state = {}
        for tag, history in tag_history.items():
            total = len(history)
            correct_cnt = sum(history)
            recent = history[-5:]
            baseline_state[tag] = {
                'concept_id': tag,
                'concept_name': self.tag_to_name.get(tag, f"Concept_{tag}"),
                'total_attempts': total,
                'correct_count': correct_cnt,
                'accuracy': float(correct_cnt / total) if total > 0 else 0.0,
                'recent_accuracy': float(sum(recent) / len(recent)) if recent else 0.0,
                'state_type': 'HISTORICAL BASELINE KNOWLEDGE STATE'
            }
        return baseline_state

    def detect_learning_gaps(self, knowledge_state: Dict[int, Dict[str, Any]],
                             mastery_threshold: float = 0.55,
                             min_attempts: int = 2) -> List[Dict[str, Any]]:
        """
        Detects learning gaps where mastery score < threshold and student has sufficient attempts.
        Ranked by lowest mastery score (greatest deficit first).
        """
        gaps = []
        for tag, info in knowledge_state.items():
            # Check mastery metric
            score = info.get('mastery_score', info.get('accuracy', 0.0))
            attempts = info.get('total_attempts', info.get('predicted_observations', 1))

            if attempts >= min_attempts and score < mastery_threshold:
                gaps.append({
                    'concept_id': tag,
                    'concept_name': info.get('concept_name', f"Concept_{tag}"),
                    'mastery': float(score),
                    'attempts': int(attempts),
                    'recent_accuracy': float(info.get('recent_accuracy', score)),
                    'gap_severity': float(mastery_threshold - score)
                })

        # Rank by lowest mastery ascending (most severe learning gap first)
        gaps.sort(key=lambda x: x['mastery'])
        return gaps

    def format_student_report(self, student_id: str,
                              model_state: Optional[Dict[int, Dict[str, Any]]] = None,
                              baseline_state: Optional[Dict[int, Dict[str, Any]]] = None,
                              gaps: Optional[List[Dict[str, Any]]] = None,
                              recommendations: Optional[List[Dict[str, Any]]] = None) -> str:
        """Formatted human-readable academic report for a learner profile."""
        lines = [
            "=" * 70,
            f"  STUDENT KNOWLEDGE STATE & PERSONALIZED RECOMMENDATION REPORT",
            f"  Student Identifier: {student_id}",
            "=" * 70
        ]

        if model_state:
            lines.append(f"\n[1] MODEL-DERIVED KNOWLEDGE STATE ({len(model_state)} concepts tracked):")
            for i, (cid, st) in enumerate(list(model_state.items())[:5]):
                lines.append(f"    - {st['concept_name']} (Tag {cid}): Model Mastery = {st['mastery_score']:.4f} ({st['predicted_observations']} interactions)")

        if baseline_state:
            lines.append(f"\n[2] HISTORICAL BASELINE KNOWLEDGE STATE ({len(baseline_state)} concepts attempted):")
            for i, (cid, st) in enumerate(list(baseline_state.items())[:5]):
                lines.append(f"    - {st['concept_name']} (Tag {cid}): Empirical Acc = {st['accuracy']:.4f} ({st['correct_count']}/{st['total_attempts']} correct, recent: {st['recent_accuracy']:.2f})")

        lines.append(f"\n[3] IDENTIFIED LEARNING GAPS (Ranked Weakest First):")
        if gaps:
            for i, g in enumerate(gaps[:5]):
                lines.append(f"    {i+1}. {g['concept_name']} (Tag {g['concept_id']}): Mastery = {g['mastery']:.4f}, Deficit = {g['gap_severity']:.4f}")
        else:
            lines.append("    No significant concept deficiencies detected (All concept masteries >= threshold).")

        if recommendations:
            lines.append(f"\n[4] TARGETED EDNET INTERVENTION RECOMMENDATIONS:")
            for i, r in enumerate(recommendations[:5]):
                lines.append(f"    {i+1}. [{r['type'].upper()}] Resource ID: {r['id']} | Topic: {r.get('concept', 'General')} | Part: {r.get('part', 'N/A')}")

        lines.append("=" * 70)
        return '\n'.join(lines)
