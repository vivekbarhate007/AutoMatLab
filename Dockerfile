FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


