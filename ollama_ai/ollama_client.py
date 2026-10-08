"""
Smart Education Project — Dedicated Ollama Client
=================================================
Robust HTTP client for interacting with the local Ollama LLM service
via official REST API endpoints (/api/tags, /api/generate, /api/chat).

Guarantees:
- Never blocks or crashes the application if Ollama is absent or unreachable.
- Uses standard Python requests library already installed in the environment.
- Configurable connection and generation timeouts.
- Clean error typing and telemetry logging without data leakage.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
import requests

from ollama_ai.ollama_config import (
    OLLAMA_ENABLED,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_TIMEOUT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    OLLAMA_MAX_TOKENS,
    OLLAMA_KEEP_ALIVE,
)

logger = logging.getLogger("SmartEducation.OllamaClient")


class OllamaClientError(Exception):
    """Base exception for all Ollama client errors."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when the Ollama service cannot be reached."""
    pass


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the requested model is not installed in Ollama."""
    pass


class OllamaTimeoutError(OllamaClientError):
    """Raised when model inference exceeds the configured timeout."""
    pass


class OllamaClient:
    """
    Dedicated client for communicating with local Ollama service.
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        connect_timeout: Optional[int] = None
    ):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_MODEL
        self.timeout = timeout or OLLAMA_TIMEOUT
        self.connect_timeout = connect_timeout or OLLAMA_CONNECT_TIMEOUT
        self._session = requests.Session()

    def is_available(self) -> bool:
        """
        Quick non-blocking probe to verify if Ollama daemon is running.
        Returns True if reachable, False otherwise.
        """
        if not OLLAMA_ENABLED:
            return False
        try:
            resp = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=self.connect_timeout
            )
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        Retrieves list of models currently pulled in local Ollama storage.
        """
        if not OLLAMA_ENABLED:
            return []
        try:
            resp = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=self.connect_timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models if m.get("name")]
            return []
        except Exception as e:
            logger.debug(f"Failed to list Ollama models: {e}")
            return []

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """
        Checks if the requested model (or default model) is installed locally.
        """
        target = (model_name or self.model).lower()
        target_base = target.split(":")[0]
        installed = [m.lower() for m in self.list_models()]
        for m in installed:
            if target == m or target_base == m.split(":")[0]:
                return True
        return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        format_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes text generation using Ollama /api/generate endpoint.
        Returns dictionary with 'response' text and performance telemetry.
        """
        if not OLLAMA_ENABLED:
            raise OllamaConnectionError("Ollama integration is disabled in configuration.")

        target_model = model or self.model
        opt = {
            "temperature": temperature if temperature is not None else OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_predict": max_tokens if max_tokens is not None else OLLAMA_MAX_TOKENS,
        }
        if options:
            opt.update(options)

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": opt
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        t_start = time.time()
        try:
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(self.connect_timeout, self.timeout)
            )
        except requests.exceptions.ConnectTimeout:
            raise OllamaConnectionError(f"Connection timed out connecting to Ollama at {self.base_url}")
        except requests.exceptions.ReadTimeout:
            raise OllamaTimeoutError(f"Ollama inference exceeded timeout of {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error communicating with Ollama: {e}")

        latency_ms = (time.time() - t_start) * 1000

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(f"Model '{target_model}' not found in local Ollama service.")
        elif resp.status_code != 200:
            raise OllamaClientError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        response_text = data.get("response", "")

        return {
            "response": response_text,
            "model": target_model,
            "latency_ms": latency_ms,
            "total_duration_ns": data.get("total_duration", 0),
            "load_duration_ns": data.get("load_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
            "status": "success"
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        format_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-turn chat completion using Ollama /api/chat endpoint.
        """
        if not OLLAMA_ENABLED:
            raise OllamaConnectionError("Ollama integration is disabled in configuration.")

        target_model = model or self.model
        opt = {
            "temperature": temperature if temperature is not None else OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_predict": max_tokens if max_tokens is not None else OLLAMA_MAX_TOKENS,
        }
        if options:
            opt.update(options)

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": opt
        }
        if format_json:
            payload["format"] = "json"

        t_start = time.time()
        try:
            resp = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(self.connect_timeout, self.timeout)
            )
        except requests.exceptions.ConnectTimeout:
            raise OllamaConnectionError(f"Connection timed out connecting to Ollama at {self.base_url}")
        except requests.exceptions.ReadTimeout:
            raise OllamaTimeoutError(f"Ollama chat inference exceeded timeout of {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error communicating with Ollama: {e}")

        latency_ms = (time.time() - t_start) * 1000

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(f"Model '{target_model}' not found in local Ollama service.")
        elif resp.status_code != 200:
            raise OllamaClientError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        message = data.get("message", {})
        response_text = message.get("content", "")

        return {
            "response": response_text,
            "message": message,
            "model": target_model,
            "latency_ms": latency_ms,
            "total_duration_ns": data.get("total_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "status": "success"
        }
