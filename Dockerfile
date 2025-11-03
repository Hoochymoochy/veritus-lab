# --- Base image ---
FROM python:3.11-slim

# --- Install deps + Ollama ---
RUN apt-get update && \
    apt-get install -y curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

# --- App setup ---
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# --- Entrypoint script ---
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start Ollama in background\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait until Ollama is ready\n\
for i in {1..30}; do\n\
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then\n\
    echo "✅ Ollama ready"\n\
    break\n\
  fi\n\
  echo "Waiting for Ollama... ($i/30)"\n\
  sleep 2\n\
done\n\
\n\
# Pre-pull models\n\
for model in nomic-embed-text mistral phi3; do\n\
  if ! ollama list | grep -q "$model"; then\n\
    echo "⬇️ Pulling $model..."\n\
    ollama pull "$model"\n\
  else\n\
    echo "✔️ $model already present"\n\
  fi\n\
done\n\
\n\
# Start Python app (foreground)\n\
exec python main.py\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# --- Ports ---
EXPOSE 4000 11434

ENTRYPOINT ["/entrypoint.sh"]
