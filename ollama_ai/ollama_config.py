"""
Smart Education Project — Ollama Generative AI Configuration
============================================================
Central configuration settings for the local Ollama LLM integration.
Enables full configurability via environment variables or direct Python imports
without hardcoding values across the codebase.

Hardware execution target:
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM)
- RAM: 16 GB
- Default Model: qwen2.5:3b (quantized 3B LLM, ~2.0 GB VRAM footprint, fast inference)
- Alternative Supported Models: llama3.2:3b, phi3:mini, mistral:7b
"""

import os
from typing import Dict, Any

# ============================================================
# Core Ollama Configuration Settings
# ============================================================

# Master toggle for Generative AI layer (system runs fine if False)
OLLAMA_ENABLED = os.getenv("SMART_EDU_OLLAMA_ENABLED", "True").lower() in ("true", "1", "yes")

# Base URL for local Ollama service (default standard port 11434)
OLLAMA_BASE_URL = os.getenv("SMART_EDU_OLLAMA_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

# Default model optimized for RTX 3050 6GB VRAM and 16GB system RAM
OLLAMA_MODEL = os.getenv("SMART_EDU_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))

# Network & Inference Timeouts (seconds)
OLLAMA_CONNECT_TIMEOUT = int(os.getenv("SMART_EDU_OLLAMA_CONNECT_TIMEOUT", "3"))
OLLAMA_TIMEOUT = int(os.getenv("SMART_EDU_OLLAMA_TIMEOUT", "35"))

# Model Generation Hyperparameters
OLLAMA_TEMPERATURE = float(os.getenv("SMART_EDU_OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_TOP_P = float(os.getenv("SMART_EDU_OLLAMA_TOP_P", "0.9"))
OLLAMA_MAX_TOKENS = int(os.getenv("SMART_EDU_OLLAMA_MAX_TOKENS", "1024"))
OLLAMA_KEEP_ALIVE = os.getenv("SMART_EDU_OLLAMA_KEEP_ALIVE", "5m")

# Recommended models list for UI selection / fallback
RECOMMENDED_MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "phi3:mini",
    "mistral:7b",
    "deepseek-r1:1.5b",
]


def get_ollama_config() -> Dict[str, Any]:
    """Returns the active Ollama configuration dictionary."""
    return {
        "enabled": OLLAMA_ENABLED,
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "connect_timeout": OLLAMA_CONNECT_TIMEOUT,
        "timeout": OLLAMA_TIMEOUT,
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
        "max_tokens": OLLAMA_MAX_TOKENS,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "recommended_models": list(RECOMMENDED_MODELS)
    }


def is_ollama_enabled() -> bool:
    """Checks if Ollama integration is enabled in config."""
    return OLLAMA_ENABLED
