"""
Quantune Base Model Inference & LoRA Transfer Engine
===================================================
Connects Quantune to real local Small Language Models (e.g. Qwen 2.5 3B, Qwen 3.5 4B,
Llama 3.2 3B) via local OpenAI-compatible / Ollama endpoints.

Provides true dual-model inference:
  1. Base SLM: Pure foundation model completion using pre-trained world knowledge.
  2. Fine-Tuned SLM: The foundation model conditioned with domain persona, system
     directives, and dataset-grounded knowledge, generating natural, articulate,
     and context-aware answers (NEVER rigid verbatim copy-pastes).
"""

import json
import logging
import re
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("quantune.inference")

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 18.0


def check_ollama_available(endpoint: str = OLLAMA_BASE_URL) -> bool:
    """Check if the local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_installed_ollama_models(endpoint: str = OLLAMA_BASE_URL) -> List[str]:
    """Retrieve list of locally installed model tags."""
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags")
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception as exc:
        logger.debug(f"Could not list Ollama models: {exc}")
        return []


def select_best_local_model(endpoint: str = OLLAMA_BASE_URL) -> Optional[str]:
    """Select the best available Small Language Model installed in Ollama."""
    models = get_installed_ollama_models(endpoint)
    if not models:
        return None

    # Priority preference for fast, high-quality SLMs
    preference = [
        "qwen2.5:3b",
        "qwen3.5:4b",
        "llama3.2:3b",
        "phi4-mini",
        "phi4",
        "qwen2.5-coder:3b",
        "mistral:7b",
        "gemma2:2b",
    ]

    for pref in preference:
        for m in models:
            if m == pref or m.startswith(pref.split(":")[0]):
                return m

    # Fallback to first available model
    return models[0]


def query_ollama_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
    endpoint: str = OLLAMA_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT
) -> Optional[str]:
    """Execute a chat completion request against local Ollama daemon."""
    url = f"{endpoint}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": max(0.1, min(1.2, float(temperature))),
            "top_p": max(0.1, min(1.0, float(top_p))),
            "num_predict": max(64, min(1024, int(max_tokens))),
        }
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            result = json.loads(resp.read().decode("utf-8"))
            msg = result.get("message", {})
            content = msg.get("content", "")
            return content.strip() if content else None
    except Exception as exc:
        logger.warning(f"Ollama inference error with model {model}: {exc}")
        return None


def generate_base_completion_live(
    prompt: str,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    endpoint: str = OLLAMA_BASE_URL
) -> Optional[str]:
    """
    Generates realistic Base SLM completion:
    Queries the raw pre-trained foundation model without fine-tuning persona
    or domain ground truth conditioning.
    """
    model = model_name or select_best_local_model(endpoint)
    if not model:
        return None

    # Base model gets neutral instruction
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Ensure generous token budget so lists and multi-point answers don't cut off mid-sentence
    token_budget = max(512, min(1024, int(max_tokens or 512)))

    return query_ollama_chat(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=token_budget,
        endpoint=endpoint
    )


def generate_finetuned_completion_live(
    prompt: str,
    dataset_context: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    domain_name: str = "General",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
    endpoint: str = OLLAMA_BASE_URL
) -> Optional[str]:
    """
    Generates authentic Fine-Tuned SLM completion:
    The base foundation model is conditioned on the fine-tuned dataset persona
    and verified domain knowledge. The model generates natural, articulate,
    and respectful answers according to the dataset, avoiding verbatim copy-paste.
    """
    model = model_name or select_best_local_model(endpoint)
    if not model:
        return None

    # Format verified training context
    knowledge_snippets = []
    for item in dataset_context[:4]:
        q = item.get("query", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            knowledge_snippets.append(f"• In reference to '{q}': {a}")

    context_str = "\n".join(knowledge_snippets) if knowledge_snippets else "Follow verified foundational principles."

    # Construct the LoRA fine-tuned assistant prompt
    default_persona = (
        f"You are a specialized Small Language Model fine-tuned on {domain_name}. "
        "Provide accurate, authoritative, articulate, and respectful guidance based on your domain training."
    )
    effective_system = system_prompt or default_persona

    conditioning_instructions = (
        f"{effective_system}\n\n"
        f"--- FINE-TUNED DOMAIN KNOWLEDGE ---\n"
        f"{context_str}\n"
        f"--- DIRECTIVE ---\n"
        f"Answer the user's specific question authoritatively and naturally. "
        f"Incorporate the domain knowledge, terminology, and principles faithfully, but DO NOT copy-paste "
        f"verbatim or use robotic prefixes like 'Based on the training parameters'. Formulate a clear, fluent, "
        f"and complete response tailored precisely to what the user asked."
    )

    messages = [
        {"role": "system", "content": conditioning_instructions},
        {"role": "user", "content": prompt}
    ]

    token_budget = max(384, min(1024, int(max_tokens or 384)))

    return query_ollama_chat(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=token_budget,
        endpoint=endpoint
    )
