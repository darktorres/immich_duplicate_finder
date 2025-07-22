FROM python:3.12-slim

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y curl build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set work directory
WORKDIR /app

# Copy only the files needed to install dependencies
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtualenvs (we're in Docker)
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

# Replace faiss-cpu with faiss-gpu-cu12
RUN pip uninstall -y faiss-cpu && \
    pip install faiss-gpu-cu12[fix-cuda]

# Copy application code
COPY . .

# Create a directory for persistent data
RUN mkdir -p /app/data
# Change file paths in the app to point to this data directory
# Note: This requires changing paths in imageDuplicate.py to use this folder, e.g. /app/data/faiss_index.bin
# For simplicity, we'll let the user mount a volume to the main directory to persist files.

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck (adjust if your page title is different)
HEALTHCHECK CMD curl -fsS http://127.0.0.1:8501 | grep -c 'title>Local Duplicate Finder</title' || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
