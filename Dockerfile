FROM python:3.11-slim

# --- System deps for build ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    libffi-dev libxml2 libxslt1.1 libxslt1-dev \
    libjpeg-dev libpng-dev \
    curl bash git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install small CPU-only torch ---
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies
COPY requirements.txt .

# --- Install remaining deps ---
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# --- Remove heavy build deps ---
RUN apt-get purge -y gcc g++ make git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 4000
CMD ["python3", "main.py"]
