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

# API Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "mistral"

# LLM Parameters
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 2048
SUMMARY_TEMPERATURE = 0.4
SUMMARY_MAX_TOKENS = 512