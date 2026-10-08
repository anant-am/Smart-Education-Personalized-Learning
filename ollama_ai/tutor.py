"""
Smart Education Project — AI Tutor & Educational Dialog Assistant
=================================================================
Interactive educational tutor grounded in real-time learner diagnostics.
Enables learners to ask questions regarding concept difficulties,
rationale behind recommendations, study schedules, and subject guidance.

Safety & Scope Enforcement:
- Strictly restricted to academic pedagogy, concept mastery, and study guidance.
- Rejects requests seeking to alter model checkpoints, execute commands,
  or extract system credentials.
- Zero-crash fallback when Ollama is offline.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple

from ollama_ai.ollama_config import OLLAMA_ENABLED
from ollama_ai.ollama_client import (
    OllamaClient, OllamaClientError, OllamaConnectionError,
    OllamaModelNotFoundError, OllamaTimeoutError
)
from ollama_ai.learner_context import LearnerContext
from ollama_ai.prompt_builder import SYSTEM_PROMPT_PEDAGOGICAL, build_tutor_prompt
from ollama_ai.fallback import fallback_tutor_response

logger = logging.getLogger("SmartEducation.AITutor")

# Patterns prohibited by safety policy
UNSAFE_PATTERNS = [
    r"\b(delete|drop|truncate|remove)\b.*\b(database|table|checkpoint|model|file)\b",
    r"\b(execute|run|powershell|cmd|bash|sh|terminal)\b",
    r"\b(password|secret|api_key|token|credential|env|private)\b",
    r"\b(override|ignore|disregard)\b.*\b(system|instruction|rules)\b",
]


def check_query_safety(query: str) -> Tuple[bool, str]:
    """
    Validates whether the user's question adheres to academic scope.
    """
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return False, (
                "Safety Notice: As an educational AI Tutor, I am restricted to explaining academic concepts, "
                "interpreting learning gaps, explaining recommendations, and building study schedules. "
                "I cannot modify system parameters, checkpoints, or execute administrative commands."
            )
    return True, ""


class AITutor:
    """
    Conversational AI Tutor grounded in learner context.
    """
    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    def is_ai_available(self) -> bool:
        """Checks if local Ollama daemon is active."""
        return self.client.is_available()

    def ask(
        self,
        context: LearnerContext,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a student query with full context grounding and safety enforcement.
        """
        # 1. Safety check
        is_safe, refusal_reason = check_query_safety(query)
        if not is_safe:
            return {
                "status": "refusal",
                "ai_available": False,
                "response": refusal_reason,
                "model": "safety-filter",
                "latency_ms": 0.0,
                "student_id": context.student_id
            }

        # 2. Availability check
        if not OLLAMA_ENABLED or not self.client.is_available():
            return fallback_tutor_response(
                context,
                query=query,
                reason="Ollama service is not running locally (default: http://localhost:11434)."
            )

        # 3. Build grounded prompt
        prompt = build_tutor_prompt(context, user_query=query, chat_history=chat_history)

        # 4. Execute inference
        try:
            res = self.client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT_PEDAGOGICAL,
                model=model
            )
            return {
                "status": "success",
                "ai_available": True,
                "response": res["response"],
                "model": res["model"],
                "latency_ms": res["latency_ms"],
                "student_id": context.student_id
            }
        except (OllamaConnectionError, OllamaTimeoutError, OllamaModelNotFoundError, OllamaClientError) as e:
            logger.warning(f"Ollama tutor inference failed: {e}. Utilizing fallback.")
            return fallback_tutor_response(context, query=query, reason=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in tutor query: {e}. Utilizing fallback.")
            return fallback_tutor_response(context, query=query, reason=f"Unexpected error: {e}")
