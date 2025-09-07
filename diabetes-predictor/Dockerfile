# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Upgrade pip, install Poetry, configure, and install dependencies without installing the project itself
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Set Streamlit config to run without browser and on 0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Command to run the app
CMD ["streamlit", "run", "app.py"]
