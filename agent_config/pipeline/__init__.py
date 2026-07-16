# agent_config/pipeline/__init__.py
"""Shared, deterministic pipeline functions used by the ADK agent tools.

Kept separate from the agent/tool wrappers so the same logic can be:
  1. Called directly by ADK tool functions (LLM-driven, in agent_config/*.py)
  2. Run standalone via run_pipeline.py (no LLM / API key required)
"""
