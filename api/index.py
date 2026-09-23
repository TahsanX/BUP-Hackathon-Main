"""Vercel serverless entry point. Same app as the Docker deployment — no
local Ollama fallback, since a serverless function has no persistent process
to hold a model in memory. LLM_PROVIDER_ORDER on Vercel must stay gemini,groq.
"""

from gridwise.api.app import app

__all__ = ["app"]
