#!/bin/bash
# Start the RL Training Service

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --quiet

# Create directories
mkdir -p models logs

# Copy environment variables from parent project if they exist
if [ -f "../.env.local" ]; then
    export $(grep -v '^#' ../.env.local | xargs)
fi

# Start the service
echo "Starting RL Training Service on http://127.0.0.1:5001"
python app.py
