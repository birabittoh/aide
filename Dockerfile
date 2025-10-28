FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY .python-version .

# Install dependencies using uv
RUN uv sync

# Copy application code
COPY *.py .

# Create directory for model cache
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "main.py"]
