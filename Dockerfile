# Trading Bot ML - Production Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for API server
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/frontend

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Create non-root user for security
RUN useradd -m -u 1000 tradingbot && chown -R tradingbot:tradingbot /app
USER tradingbot

# Default command
CMD ["python", "api_server.py"]