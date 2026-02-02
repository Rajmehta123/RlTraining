FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# 1. Install PyTorch CPU first (to avoid downloading the huge GPU version)
# We use python 3.11 because ML wheels are often not available for the newest python versions immediately
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies
# Increase timeout for slow installs
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Create directories for models and logs
RUN mkdir -p models logs

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Expose the port
EXPOSE 5001

# Run gunicorn
# - w 1: 1 worker process (sufficient for the threading model used in app.py)
# - threads 8: support multiple concurrent requests
# - timeout 120: allow longer request processing if needed
# - bind 0.0.0.0:$PORT: bind to the port provided by the environment
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-5001} --workers 1 --threads 8 --timeout 120 app:app"
