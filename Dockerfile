# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv configuration files
COPY pyproject.toml ./
COPY uv.lock ./

# Install uv
RUN pip install uv

# Install Python dependencies using uv
RUN uv sync --frozen

# Copy project files
COPY project_files/ ./project_files/
COPY README.md ./
COPY sync_config.ps1 ./
COPY sync_docker.ps1 ./
COPY sync_linting.ps1 ./
COPY sync_automation.ps1 ./

# Create necessary directories
RUN mkdir -p project_files/models
RUN mkdir -p project_files/data
RUN mkdir -p project_files/logs

# Set permissions
RUN chmod +x sync_*.ps1

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command to run the FastAPI app
CMD ["uv", "run", "python", "-m", "uvicorn", "project_files.src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 