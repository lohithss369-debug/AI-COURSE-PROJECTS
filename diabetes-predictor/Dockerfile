# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY pyproject.toml poetry.lock* ./
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . .

# Expose port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
