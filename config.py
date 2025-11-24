"""
Configuration settings for the legal AI system.
"""
import os
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# OpenAI Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "project": os.getenv("OPENAI_PROJECT_ID"),
    "organization": os.getenv("OPENAI_ORG_ID")  # Optional
}

# Model Configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Still using sentence-transformers for embeddings
LLM_MODEL = "gpt-4o-mini"  # OpenAI model (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)

# LLM Parameters
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 2048
SUMMARY_TEMPERATURE = 0.4
SUMMARY_MAX_TOKENS = 512

# Legacy Ollama URL (kept for backward compatibility if needed)
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")