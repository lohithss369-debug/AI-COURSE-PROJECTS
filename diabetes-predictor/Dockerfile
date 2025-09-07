# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only dependency files first for caching
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi

# Copy the rest of the app
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
