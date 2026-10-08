"""
Smart Education Project — Ollama Generative AI Module
======================================================
Additive, modular Generative AI interpretation and tutoring layer.
Interprets structured knowledge states, explains learning gaps, explains
recommendations, constructs personalized study schedules, and provides
an interactive AI tutor grounded strictly in real ML metrics.

Components:
- ollama_config: Centralized settings and hardware configurations.
- ollama_client: Direct HTTP communication with local Ollama service.
- learner_context: Clean data representation mapping real ML outputs.
- prompt_builder: Programmatic, grounded pedagogical prompts.
- response_generator: High-level generation orchestrator with graceful fallback.
- tutor: Safe, educational conversational tutoring assistant.
- fallback: Deterministic, rule-based responses when Ollama is offline.
"""

from ollama_ai.ollama_config import (
    OLLAMA_ENABLED,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_TEMPERATURE,
    get_ollama_config,
    is_ollama_enabled,
)
from ollama_ai.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from ollama_ai.learner_context import LearnerContext, build_learner_context
from ollama_ai.response_generator import ResponseGenerator
from ollama_ai.tutor import AITutor

__all__ = [
    "OLLAMA_ENABLED",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT",
    "OLLAMA_TEMPERATURE",
    "get_ollama_config",
    "is_ollama_enabled",
    "OllamaClient",
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "LearnerContext",
    "build_learner_context",
    "ResponseGenerator",
    "AITutor",
]
