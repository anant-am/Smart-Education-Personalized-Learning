"""
Smart Education Project — Programmatic Prompt Engineering
==========================================================
Centralized, versioned prompt templates grounding LLM generations
strictly within actual ML predictions, concept masteries, and authentic
EdNet recommendations.

Invariant:
- Prompts forbid hallucination of non-existent scores or learning resources.
- Prompts adapt to the learner's attention state (Attentive, Partially Attentive, Inattentive).
"""

from typing import List, Dict, Any, Optional
from ollama_ai.learner_context import LearnerContext

PROMPT_VERSION = "1.0.0"

SYSTEM_PROMPT_PEDAGOGICAL = """You are an academic AI Pedagogical Assistant integrated into the Smart Education System.
Your responsibility is to interpret machine learning knowledge-tracing outputs, explain diagnosed concept deficiencies, explain why specific learning materials were recommended, construct personalized study schedules, and act as a supportive AI tutor.

STRICT OPERATIONAL DIRECTIVES:
1. Ground all feedback strictly on the provided real machine learning predictions, concept masteries, and authentic EdNet recommendations.
2. NEVER invent, extrapolate, or hallucinate mastery scores, model accuracies, test histories, or educational resources not listed in the context.
3. If attention telemetry is present (Attentive, Partially Attentive, Inattentive), reflect appropriate pedagogical pacing (e.g., bite-sized video lectures and worked explanations for lower attention; active recall and practice questions for high attention).
4. Maintain a supportive, encouraging, professional academic tone.
5. If certain data is marked 'Unavailable', acknowledge its absence rather than guessing.
"""


def build_learning_gap_prompt(context: LearnerContext) -> str:
    """
    Constructs prompt requesting natural-language explanation of diagnosed concept deficits.
    """
    ctx_text = context.format_for_prompt()
    prompt = f"""Below is the verified diagnostic performance profile for a learner produced by the Smart Education system:

====================== LEARNER CONTEXT ======================
{ctx_text}
=============================================================

TASK:
Provide a clear, pedagogically structured explanation of this learner's knowledge state and concept deficiencies:

1. Proficiency Overview: Briefly interpret the multi-model neural consensus and overall readiness.
2. Core Deficiencies Analysis: For each diagnosed critical learning gap (mastery < 60%), explain what the deficit indicates and why it represents a foundational priority.
3. Remedial Action: Explain what foundational concepts or skills need immediate reinforcement.

Keep the response clear, structured, and strictly faithful to the provided context. Do NOT fabricate any statistics."""
    return prompt


def build_recommendation_explanation_prompt(context: LearnerContext) -> str:
    """
    Constructs prompt explaining why specific authentic EdNet resources were selected and ranked.
    """
    ctx_text = context.format_for_prompt()
    prompt = f"""Below is the verified diagnostic profile and recommended learning resources for a learner:

====================== LEARNER CONTEXT ======================
{ctx_text}
=============================================================

TASK:
Explain to the learner WHY these specific learning materials were selected for their study plan:

1. Strategic Rationale: Explain why the engine selected these specific items targeting their diagnosed weak concepts.
2. Pedagogical Sequencing: Explain the progression from Conceptual Lectures (Foundations) -> Worked Explanations (Error Remediation) -> Practice Questions (Active Retrieval).
3. Engagement Adaptation: If multimodal attention data is present ({context.attention_state or 'Neutral'}), explain how their current attention level informs the recommended study rhythm and video duration.

Refer only to the specific Resource IDs and Concept IDs provided. Do NOT invent new resources."""
    return prompt


def build_study_plan_prompt(context: LearnerContext, target_hours: float = 2.0) -> str:
    """
    Constructs prompt for generating a structured, actionable study schedule.
    """
    ctx_text = context.format_for_prompt()
    prompt = f"""Below is the diagnostic learning context and recommended materials for a learner:

====================== LEARNER CONTEXT ======================
{ctx_text}
=============================================================

TASK:
Create a realistic, structured {target_hours:.1f}-hour personalized study plan for this student:

1. Schedule Breakdown: Organize the session into distinct phases (e.g., Phase 1: Conceptual Foundations, Phase 2: Guided Examples, Phase 3: Active Practice & Self-Check).
2. Resource Mapping: Explicitly assign the recommended items (by ID and Concept #) to each phase.
3. Attention & Pacing Guidance: Consider the learner's attention state ({context.attention_state or 'Neutral'}) to suggest optimal break intervals and review strategies.
4. Measurable Milestone: State one concrete learning goal the student will achieve upon completing this plan.

Keep the plan realistic, encouraging, and strictly aligned with the recommended items."""
    return prompt


def build_attention_explanation_prompt(context: LearnerContext) -> str:
    """
    Constructs prompt specifically explaining multimodal attention analysis in relation to learning.
    """
    ctx_text = context.format_for_prompt()
    prompt = f"""Below is the combined multimodal attention telemetry and knowledge profile for a learner:

====================== LEARNER CONTEXT ======================
{ctx_text}
=============================================================

TASK:
Provide an explanation of the learner's real-time engagement and its pedagogical implications:

1. Engagement Diagnosis: Interpret the detected attention state ({context.attention_state}) and confidence score based on the facial/acoustic sensory signals.
2. Cognitive Load & Fatigue: Explain what this engagement pattern means for their current cognitive capacity.
3. Adaptive Pedagogical Strategy: Explain why the learning sequence has been adjusted (e.g., shorter video lectures vs active problem-solving) to match this attention state.
4. Learner Advice: Provide practical, positive guidance for maintaining or restoring peak study focus."""
    return prompt


def build_tutor_prompt(
    context: LearnerContext,
    user_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Constructs prompt for conversational AI Tutor grounded in learner context.
    """
    ctx_text = context.format_for_prompt()
    history_lines = []
    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "No prior conversation turns."

    prompt = f"""You are the student's personal AI Tutor for Smart Education. You have access to their authentic performance and diagnostic metrics:

====================== LEARNER CONTEXT ======================
{ctx_text}
=============================================================

CONVERSATION HISTORY:
{history_str}

STUDENT QUESTION:
"{user_query}"

TUTOR INSTRUCTIONS:
- Directly answer the student's question using their actual diagnostic metrics, weak concepts, and recommendations where relevant.
- Explain concepts clearly, with helpful examples or analogies.
- If the student asks why they are struggling or why something was recommended, cite their actual mastery and deficit data.
- Never invent test results or alter system parameters.
- Provide a concise, supportive, educational response."""
    return prompt
