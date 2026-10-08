"""
Smart Education Project — Resilient Rule-Based Fallback System
==============================================================
Provides guaranteed, zero-crash deterministic fallback explanations
whenever the Ollama Generative AI daemon is offline, uninstalled,
unreachable, or experiencing timeouts.

Principles:
1. NEVER crash or throw unhandled exceptions to the caller.
2. Clearly mark outputs with `ai_available = False` and the exact status reason.
3. Generate educational, helpful, deterministic insights directly from
   the real ML metrics (Knowledge State, Learning Gaps, Attention, Recommendations).
"""

from typing import Dict, Any, List, Optional
from ollama_ai.learner_context import LearnerContext


def fallback_explain_learning_gaps(
    context: LearnerContext,
    reason: str = "Ollama Generative AI service is currently offline or unreachable."
) -> Dict[str, Any]:
    """
    Deterministic rule-based explanation of diagnosed concept deficits.
    """
    gaps = context.learning_gaps
    consensus = context.ensemble_consensus
    consensus_str = f"{consensus*100:.1f}%" if consensus is not None else "N/A"

    if not gaps:
        text = (
            f"**Diagnostic Summary (ML System)**\n\n"
            f"The 5-model neural ensemble estimates a next-question readiness probability of **{consensus_str}**.\n\n"
            f"✅ **No Critical Knowledge Deficiencies:** All evaluated concept mastery levels meet or exceed the "
            f"60.0% remediation threshold. Continue with advanced practice and comprehensive question bundles."
        )
    else:
        gap_items = []
        for idx, g in enumerate(gaps[:4], 1):
            cid = g.get("concept_id", g.get("concept", "N/A"))
            m = g.get("mastery_score", g.get("mastery", 0.0))
            att = g.get("attempts", g.get("total_attempts", 1))
            sev = g.get("gap_severity", max(0.0, 0.60 - m))
            gap_items.append(
                f"- **Priority {idx} — Concept #{cid}**: Estimated mastery is **{m*100:.1f}%** "
                f"(Deficit gap: {sev*100:.1f}% across {att} evaluated attempts). Remedial theory and worked solutions are required."
            )

        text = (
            f"**Diagnostic Summary (ML System)**\n\n"
            f"The 5-model neural ensemble estimates a next-question readiness probability of **{consensus_str}**.\n\n"
            f"⚠️ **{len(gaps)} Critical Knowledge Deficiencies Detected:**\n"
            + "\n".join(gap_items) + "\n\n"
            f"**Remedial Guidance:** Prioritize reviewing foundational concepts with lowest mastery scores first. "
            f"Completing targeted video lectures and step-by-step worked explanations will stabilize foundational retrieval."
        )

    return {
        "status": "fallback",
        "ai_available": False,
        "reason": reason,
        "explanation": text,
        "student_id": context.student_id,
        "total_gaps": len(gaps),
    }


def fallback_explain_recommendations(
    context: LearnerContext,
    reason: str = "Ollama Generative AI service is currently offline or unreachable."
) -> Dict[str, Any]:
    """
    Deterministic rule-based explanation of why specific EdNet items were recommended.
    """
    recs = context.recommended_resources
    att_state = context.attention_state

    if not recs:
        text = (
            "**Recommendation Logic (Rule-Based)**\n\n"
            "No specific remedial recommendations are active. All evaluated concepts are above "
            "the 60.0% threshold, or all candidate items for weak concepts have already been completed."
        )
    else:
        rec_items = []
        for idx, r in enumerate(recs[:5], 1):
            rtype = str(r.get("type", "Resource")).capitalize()
            rid = r.get("id", "N/A")
            cid = r.get("concept", "N/A")
            pacing = f" • *{r.get('pacing_tag')}*" if "pacing_tag" in r else ""
            dur = f" ({r.get('duration_sec')}s)" if r.get("duration_sec") and r.get("duration_sec") > 0 else ""
            rec_items.append(f"{idx}. **[{rtype}]** `{rid}` targeting **Concept #{cid}**{dur}{pacing}")

        att_note = ""
        if att_state:
            att_note = (
                f"\n\n**Engagement Pacing ({att_state}):** "
                f"The sequence is paced according to your real-time attention signals to prevent cognitive overload."
            )

        text = (
            f"**Why These Resources Were Selected (Curriculum Graph & Recommender Engine)**\n\n"
            f"The recommendation engine applies pedagogical hierarchy targeting your diagnosed weaknesses:\n"
            f"1. **Video Lectures (Conceptual Theory):** Introduced first to establish schema and terminology.\n"
            f"2. **Worked Explanations (Error Remediation):** Provide concrete step-by-step problem deconstructions.\n"
            f"3. **Practice Questions (Active Retrieval):** Consolidate memory and verify mastery progression.\n\n"
            f"**Curated Learning Sequence:**\n"
            + "\n".join(rec_items)
            + att_note
        )

    return {
        "status": "fallback",
        "ai_available": False,
        "reason": reason,
        "explanation": text,
        "student_id": context.student_id,
        "recommended_count": len(recs),
    }


def fallback_study_plan(
    context: LearnerContext,
    target_hours: float = 2.0,
    reason: str = "Ollama Generative AI service is currently offline or unreachable."
) -> Dict[str, Any]:
    """
    Deterministic rule-based personalized study schedule.
    """
    recs = context.recommended_resources
    att = context.attention_state or "Standard"

    lectures = [r for r in recs if r.get("type") == "lecture"]
    explanations = [r for r in recs if r.get("type") == "explanation"]
    questions = [r for r in recs if r.get("type") == "question"]

    # Calculate time partitions
    total_mins = int(target_hours * 60)
    phase1_mins = max(15, int(total_mins * 0.35))
    phase2_mins = max(15, int(total_mins * 0.35))
    phase3_mins = max(10, total_mins - phase1_mins - phase2_mins)

    plan_steps = [
        {
            "phase": "Phase 1: Conceptual Foundations",
            "duration_minutes": phase1_mins,
            "target": "Build underlying concept schema",
            "resources": [f"[{r.get('type').upper()}] {r.get('id')} (Concept #{r.get('concept')})" for r in lectures[:2]] or ["Review fundamental concept notes"],
            "guidance": "Watch at regular speed; take brief structured notes on formulas and rules."
        },
        {
            "phase": "Phase 2: Worked Example Deconstruction",
            "duration_minutes": phase2_mins,
            "target": "Analyze step-by-step problem solutions",
            "resources": [f"[{r.get('type').upper()}] {r.get('id')} (Concept #{r.get('concept')})" for r in explanations[:2]] or ["Review worked practice solutions"],
            "guidance": "Follow each solution step carefully; identify typical distractor choices."
        },
        {
            "phase": "Phase 3: Active Retrieval & Practice",
            "duration_minutes": phase3_mins,
            "target": "Reinforce memory through testing",
            "resources": [f"[{r.get('type').upper()}] {r.get('id')} (Concept #{r.get('concept')})" for r in questions[:3]] or ["Complete 5 targeted practice questions"],
            "guidance": "Attempt questions under timed conditions (approx. 45-60s per item)."
        }
    ]

    break_advice = "Take a 5-minute cognitive break between Phase 1 and Phase 2."
    if att == "Inattentive":
        break_advice = "Attention level is low: Take a 5-minute break every 20 minutes to restore focus."
    elif att == "Partially Attentive":
        break_advice = "Attention level is moderate: Incorporate a 5-minute micro-break halfway through the session."

    formatted_text = (
        f"**Personalized Study Schedule ({target_hours:.1f} Hours)**\n\n"
        f"**Focus Strategy:** Grounded in diagnosed concept gaps and paced for **{att}** attention level.\n\n"
    )
    for p in plan_steps:
        formatted_text += f"### {p['phase']} ({p['duration_minutes']} mins)\n"
        formatted_text += f"- **Goal:** {p['target']}\n"
        formatted_text += f"- **Materials:** {', '.join(p['resources'])}\n"
        formatted_text += f"- **Pacing Note:** {p['guidance']}\n\n"
    formatted_text += f"💡 **Cognitive Focus Tip:** {break_advice}"

    return {
        "status": "fallback",
        "ai_available": False,
        "reason": reason,
        "study_plan": plan_steps,
        "explanation": formatted_text,
        "student_id": context.student_id,
        "target_hours": target_hours
    }


def fallback_attention_explanation(
    context: LearnerContext,
    reason: str = "Ollama Generative AI service is currently offline or unreachable."
) -> Dict[str, Any]:
    """
    Deterministic rule-based explanation of video + audio attention telemetry.
    """
    att_class = context.attention_state or "Neutral / Unknown"
    conf = context.attention_confidence or 0.0

    if att_class == "Inattentive":
        strategy = "Cognitive Relief & Foundations"
        implication = "Sensory signals indicate eye gaze divergence and reduced response engagement."
        pacing = "Prioritizing concise video lectures under 5 minutes and step-by-step solutions to prevent overload."
    elif att_class == "Partially Attentive":
        strategy = "Guided Pacing"
        implication = "Sensory telemetry indicates moderate visual focus with periodic attentional fluctuations."
        pacing = "Balancing worked solutions with single practice questions to maintain dynamic cognitive flow."
    else:
        strategy = "Accelerated Mastery"
        implication = "Optimal face orientation, steady gaze fixation, and active acoustic alertness detected."
        pacing = "Prioritizing challenging practice questions and comprehensive deep-dive video materials."

    text = (
        f"**Multimodal Attention Telemetry Analysis**\n\n"
        f"- **Diagnosed Engagement State:** **{att_class}** ({conf*100:.1f}% confidence)\n"
        f"- **Biometric Telemetry Interpretation:** {implication}\n"
        f"- **Adaptive Remedial Strategy:** {strategy}\n"
        f"- **Pedagogical Action:** {pacing}"
    )

    return {
        "status": "fallback",
        "ai_available": False,
        "reason": reason,
        "explanation": text,
        "attention_state": att_class,
        "confidence": conf
    }


def fallback_tutor_response(
    context: LearnerContext,
    query: str,
    reason: str = "Ollama Generative AI service is currently offline or unreachable."
) -> Dict[str, Any]:
    """
    Deterministic rule-based response to learner queries.
    """
    q_lower = query.lower()
    gaps = context.learning_gaps
    consensus = context.ensemble_consensus
    consensus_str = f"{consensus*100:.1f}%" if consensus is not None else "N/A"

    if "gap" in q_lower or "weak" in q_lower or "struggl" in q_lower or "concept" in q_lower:
        if gaps:
            top_gap = gaps[0]
            cid = top_gap.get("concept_id", top_gap.get("concept", "N/A"))
            m = top_gap.get("mastery_score", top_gap.get("mastery", 0.0))
            ans = (
                f"Based on your neural knowledge state evaluation, your primary critical weakness is "
                f"**Concept #{cid}**, with an estimated mastery of **{m*100:.1f}%**.\n\n"
                f"You have {len(gaps)} total concept deficits below the 60% threshold. "
                f"Your overall next-question readiness consensus across 5 models is **{consensus_str}**.\n\n"
                f"We recommend reviewing the foundational video lectures and worked explanations assigned to Concept #{cid}."
            )
        else:
            ans = f"Your diagnostics show strong overall mastery across all evaluated concepts (Consensus: {consensus_str}). No critical deficits below 60% were identified."

    elif "recommend" in q_lower or "why" in q_lower or "resource" in q_lower:
        recs = context.recommended_resources
        if recs:
            first_rec = recs[0]
            ans = (
                f"Resources are recommended to target your lowest-mastery concepts in pedagogical order:\n"
                f"1. Foundations first: Video lectures introduce concepts.\n"
                f"2. Worked explanations demonstrate solution methods.\n"
                f"3. Active practice questions test retention.\n\n"
                f"Your top priority resource is **[{first_rec.get('type').upper()}] `{first_rec.get('id')}`** "
                f"addressing Concept **#{first_rec.get('concept')}**."
            )
        else:
            ans = "No active recommendations are currently queued for your profile."

    elif "study plan" in q_lower or "schedule" in q_lower or "today" in q_lower:
        ans = (
            f"Here is a recommended 3-phase study routine based on your diagnostic state:\n"
            f"• **Phase 1 (30m):** Foundational review of Concept #{gaps[0].get('concept_id', 'N/A') if gaps else '18'} (Lecture)\n"
            f"• **Phase 2 (30m):** Worked explanation deconstruction\n"
            f"• **Phase 3 (30m):** Active retrieval quiz (5 practice questions)\n"
            f"• Take a 5-minute break between each phase."
        )

    else:
        ans = (
            f"Hello! I am your AI learning assistant. "
            f"I have access to your performance metrics (Consensus Readiness: **{consensus_str}**, "
            f"Diagnosed Deficits: **{len(gaps)}**).\n\n"
            f"You can ask me:\n"
            f"- 'What are my top learning gaps?'\n"
            f"- 'Why was this video lecture recommended to me?'\n"
            f"- 'Give me a study plan for today.'"
        )

    return {
        "status": "fallback",
        "ai_available": False,
        "reason": reason,
        "response": ans,
        "student_id": context.student_id
    }
