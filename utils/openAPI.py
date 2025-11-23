from openai import AsyncOpenAI
import os

openai = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)
