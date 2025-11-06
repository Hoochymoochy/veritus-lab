# --- Base image (GPU ready) ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- System setup ---
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl ca-certificates bash && \
    rm -rf /var/lib/apt/lists/*

# --- Install Ollama ---
RUN curl -fsSL https://ollama.com/install.sh | bash

# --- App setup ---
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./

# Install GPU-compatible PyTorch + rest of deps
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# --- Entrypoint script ---
RUN cat << 'EOF' > /entrypoint.sh
#!/bin/bash
set -e

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama ready"
    break
  fi
  echo "Waiting for Ollama... ($i/30)"
  sleep 2
done

# Pre-pull models
for model in mistral; do
  if ! ollama list | grep -q "$model"; then
    echo "⬇️ Pulling $model..."
    ollama pull "$model"
  else
    echo "✔️ $model already present"
  fi
done

# Launch the backend
exec python3 main.py
EOF

RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
RUN chmod +x /entrypoint.sh

# --- Ports ---
EXPOSE 4000 11434

# --- Run ---
ENTRYPOINT ["/entrypoint.sh"]
