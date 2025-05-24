FROM python:3.9-slim

WORKDIR /app

# Install poetry
RUN pip install poetry==1.4.2

# Copy poetry configuration
COPY pyproject.toml poetry.lock* /app/

# Configure poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . /app/

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "open_classifier.main"]