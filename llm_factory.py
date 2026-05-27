"""
LLM Factory for Multi-Provider Support

Supports switching between OpenAI, Google Gemini, and Groq
for ultra-low TTFT (Time to First Token).

Provider Performance:
- Groq: 100-200ms TTFT (fastest)
- Google Gemini: 300-400ms TTFT
- OpenAI: 700ms TTFT
"""

import logging
import os
from typing import Optional

from config import LLMConfig

logger = logging.getLogger("llm-factory")


def create_llm(config: LLMConfig, agent_doc: dict):
    """
    Create LLM instance based on provider configuration.
    
    Args:
        config: LLM configuration with provider settings
        agent_doc: Agent document from MongoDB (may override model)
    
    Returns:
        LLM instance for the selected provider
    
    Raises:
        ValueError: If provider is unknown
        ImportError: If provider plugin not installed
    """
    # Allow environment variable override
    provider = os.getenv("LLM_PROVIDER", config.provider).lower()
    
    logger.info(f"Creating LLM with provider: {provider}")
    
    try:
        if provider == "openai":
            return _create_openai_llm(config, agent_doc)
        elif provider in ["google", "gemini"]:
            return _create_google_llm(config, agent_doc)
        elif provider == "groq":
            return _create_groq_llm(config, agent_doc)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    except Exception as e:
        logger.error(f"Failed to create {provider} LLM: {e}")
        
        # Try fallback if enabled
        if config.enable_fallback and provider != config.fallback_provider:
            logger.warning(f"Falling back to {config.fallback_provider}")
            return _create_fallback_llm(config, agent_doc)
        else:
            raise


def _create_openai_llm(config: LLMConfig, agent_doc: dict):
    """Create OpenAI LLM instance."""
    try:
        from livekit.plugins import openai
    except ImportError:
        raise ImportError("OpenAI plugin not installed. Run: pip install livekit-plugins-openai")
    
    # Use config model by default, only override if agent_doc has an OpenAI model
    model = config.openai_model
    if "llm_model" in agent_doc:
        agent_model = agent_doc["llm_model"]
        # Only use agent_doc model if it's an OpenAI model
        if any(x in agent_model.lower() for x in ["gpt", "o1", "o3"]):
            model = agent_model
    
    llm = openai.LLM(
        model=model,
        temperature=config.openai_temperature,
        max_completion_tokens=config.openai_max_tokens,
    )
    
    logger.info(f"OpenAI LLM created: {model}")
    return llm


def _create_google_llm(config: LLMConfig, agent_doc: dict):
    """Create Google Gemini LLM instance."""
    try:
        from livekit.plugins import google
    except ImportError:
        raise ImportError("Google plugin not installed. Run: pip install livekit-plugins-google")
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Use config model by default, only override if agent_doc has a Gemini model
    model = config.google_model
    if "llm_model" in agent_doc:
        agent_model = agent_doc["llm_model"]
        # Only use agent_doc model if it's a Gemini model
        if "gemini" in agent_model.lower():
            model = agent_model
    
    llm = google.LLM(
        model=model,
        temperature=config.google_temperature,
        max_output_tokens=config.google_max_tokens,
    )
    
    logger.info(f"Google Gemini LLM created: {model}")
    return llm


def _create_groq_llm(config: LLMConfig, agent_doc: dict):
    """Create Groq LLM instance (fastest)."""
    try:
        from livekit.plugins import groq
    except ImportError:
        raise ImportError("Groq plugin not installed. Run: pip install livekit-plugins-groq")
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Use config model by default, only override if agent_doc has a Groq/Llama model
    model = config.groq_model
    if "llm_model" in agent_doc:
        agent_model = agent_doc["llm_model"]
        # Only use agent_doc model if it's a Groq-compatible model
        if any(x in agent_model.lower() for x in ["llama", "mixtral", "groq"]):
            model = agent_model
    
    llm = groq.LLM(
        model=model,
        temperature=config.groq_temperature,
        max_tokens=config.groq_max_tokens,
    )
    
    logger.info(f"Groq LLM created: {model} (ultra-fast TTFT)")
    return llm


def _create_fallback_llm(config: LLMConfig, agent_doc: dict):
    """Create fallback LLM (usually OpenAI)."""
    fallback_provider = config.fallback_provider.lower()
    
    logger.info(f"Creating fallback LLM: {fallback_provider}")
    
    if fallback_provider == "openai":
        return _create_openai_llm(config, agent_doc)
    elif fallback_provider in ["google", "gemini"]:
        return _create_google_llm(config, agent_doc)
    elif fallback_provider == "groq":
        return _create_groq_llm(config, agent_doc)
    else:
        # Ultimate fallback - OpenAI
        logger.warning(f"Unknown fallback provider {fallback_provider}, using OpenAI")
        return _create_openai_llm(config, agent_doc)


def get_provider_info(config: LLMConfig) -> dict:
    """
    Get information about the configured LLM provider.
    
    Returns:
        Dictionary with provider details
    """
    provider = os.getenv("LLM_PROVIDER", config.provider).lower()
    
    provider_info = {
        "openai": {
            "name": "OpenAI",
            "model": config.openai_model,
            "expected_ttft_ms": 700,
            "description": "Reliable, good quality",
        },
        "google": {
            "name": "Google Gemini",
            "model": config.google_model,
            "expected_ttft_ms": 350,
            "description": "Fast, balanced performance",
        },
        "groq": {
            "name": "Groq",
            "model": config.groq_model,
            "expected_ttft_ms": 150,
            "description": "Ultra-fast inference",
        },
    }
    
    return provider_info.get(provider, provider_info["openai"])
