"""
Smart Education Project — Generative AI Response Generator
===========================================================
High-level orchestrator mediating between structured learner context,
programmatic prompt generation, local Ollama execution, and guaranteed
deterministic fallbacks.

Guarantees:
- Safe execution: NEVER raises unhandled exceptions to UI or API layers.
- Automatic degradation: seamless fallback to deterministic rule-based output
  if Ollama is offline, slow, or returning errors.
- Telemetry: captures inference latency, model used, and token stats.
"""

import time
import logging
from typing import Dict, Any, Optional

from ollama_ai.ollama_config import (
    OLLAMA_ENABLED, OLLAMA_MODEL, get_ollama_config
)
from ollama_ai.ollama_client import (
    OllamaClient, OllamaClientError, OllamaConnectionError,
    OllamaModelNotFoundError, OllamaTimeoutError
)
from ollama_ai.learner_context import LearnerContext
from ollama_ai.prompt_builder import (
    SYSTEM_PROMPT_PEDAGOGICAL,
    build_learning_gap_prompt,
    build_recommendation_explanation_prompt,
    build_study_plan_prompt,
    build_attention_explanation_prompt
)
from ollama_ai.fallback import (
    fallback_explain_learning_gaps,
    fallback_explain_recommendations,
    fallback_study_plan,
    fallback_attention_explanation
)

logger = logging.getLogger("SmartEducation.ResponseGenerator")


class ResponseGenerator:
    """
    Orchestrates LLM inference with graceful fallback guarantees.
    """
    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    def is_ai_available(self) -> bool:
        """Checks if local Ollama service is reachable."""
        return self.client.is_available()

    def explain_learning_gaps(
        self,
        context: LearnerContext,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates contextual natural-language explanation of diagnosed concept weaknesses.
        """
        if not OLLAMA_ENABLED or not self.client.is_available():
            return fallback_explain_learning_gaps(
                context,
                reason="Ollama service is not running locally (default: http://localhost:11434)."
            )

        prompt = build_learning_gap_prompt(context)
        try:
            res = self.client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT_PEDAGOGICAL,
                model=model
            )
            return {
                "status": "success",
                "ai_available": True,
                "explanation": res["response"],
                "model": res["model"],
                "latency_ms": res["latency_ms"],
                "student_id": context.student_id,
                "total_gaps": len(context.learning_gaps)
            }
        except (OllamaConnectionError, OllamaTimeoutError, OllamaModelNotFoundError, OllamaClientError) as e:
            logger.warning(f"Ollama generation failed for learning gaps: {e}. Utilizing fallback.")
            return fallback_explain_learning_gaps(context, reason=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in explain_learning_gaps: {e}. Utilizing fallback.")
            return fallback_explain_learning_gaps(context, reason=f"Unexpected error: {e}")

    def explain_recommendations(
        self,
        context: LearnerContext,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates contextual natural-language explanation of why specific EdNet items were recommended.
        """
        if not OLLAMA_ENABLED or not self.client.is_available():
            return fallback_explain_recommendations(
                context,
                reason="Ollama service is not running locally (default: http://localhost:11434)."
            )

        prompt = build_recommendation_explanation_prompt(context)
        try:
            res = self.client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT_PEDAGOGICAL,
                model=model
            )
            return {
                "status": "success",
                "ai_available": True,
                "explanation": res["response"],
                "model": res["model"],
                "latency_ms": res["latency_ms"],
                "student_id": context.student_id,
                "recommended_count": len(context.recommended_resources)
            }
        except (OllamaConnectionError, OllamaTimeoutError, OllamaModelNotFoundError, OllamaClientError) as e:
            logger.warning(f"Ollama generation failed for recommendations: {e}. Utilizing fallback.")
            return fallback_explain_recommendations(context, reason=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in explain_recommendations: {e}. Utilizing fallback.")
            return fallback_explain_recommendations(context, reason=f"Unexpected error: {e}")

    def generate_study_plan(
        self,
        context: LearnerContext,
        target_hours: float = 2.0,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a personalized, structured study plan integrating diagnosed weaknesses.
        """
        if not OLLAMA_ENABLED or not self.client.is_available():
            return fallback_study_plan(
                context,
                target_hours=target_hours,
                reason="Ollama service is not running locally (default: http://localhost:11434)."
            )

        prompt = build_study_plan_prompt(context, target_hours=target_hours)
        try:
            res = self.client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT_PEDAGOGICAL,
                model=model
            )
            return {
                "status": "success",
                "ai_available": True,
                "explanation": res["response"],
                "model": res["model"],
                "latency_ms": res["latency_ms"],
                "student_id": context.student_id,
                "target_hours": target_hours
            }
        except (OllamaConnectionError, OllamaTimeoutError, OllamaModelNotFoundError, OllamaClientError) as e:
            logger.warning(f"Ollama generation failed for study plan: {e}. Utilizing fallback.")
            return fallback_study_plan(context, target_hours=target_hours, reason=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in generate_study_plan: {e}. Utilizing fallback.")
            return fallback_study_plan(context, target_hours=target_hours, reason=f"Unexpected error: {e}")

    def explain_multimodal_attention(
        self,
        context: LearnerContext,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates pedagogical explanation of video + audio attention telemetry.
        """
        if not OLLAMA_ENABLED or not self.client.is_available():
            return fallback_attention_explanation(
                context,
                reason="Ollama service is not running locally (default: http://localhost:11434)."
            )

        prompt = build_attention_explanation_prompt(context)
        try:
            res = self.client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT_PEDAGOGICAL,
                model=model
            )
            return {
                "status": "success",
                "ai_available": True,
                "explanation": res["response"],
                "model": res["model"],
                "latency_ms": res["latency_ms"],
                "attention_state": context.attention_state,
                "confidence": context.attention_confidence
            }
        except (OllamaConnectionError, OllamaTimeoutError, OllamaModelNotFoundError, OllamaClientError) as e:
            logger.warning(f"Ollama generation failed for attention explanation: {e}. Utilizing fallback.")
            return fallback_attention_explanation(context, reason=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in explain_multimodal_attention: {e}. Utilizing fallback.")
            return fallback_attention_explanation(context, reason=f"Unexpected error: {e}")
