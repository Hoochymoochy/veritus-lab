# --- Base image ---
FROM python:3.11-slim

# --- System setup ---
RUN apt-get update && \
    apt-get install -y curl bash && \
    rm -rf /var/lib/apt/lists/*

# --- App setup ---
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# --- Expose your backend port ---
EXPOSE 4000

# --- Run the backend ---
CMD ["python3", "main.py"]
