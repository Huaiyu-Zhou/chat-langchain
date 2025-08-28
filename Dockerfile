FROM python:3.11-buster

# Install Poetry
RUN pip install poetry==1.5.1

# Configure Poetry
RUN poetry config virtualenvs.create false

WORKDIR /app

# Copy project metadata first (better cache usage)
COPY pyproject.toml poetry.lock* README.md* /app/

# Install dependencies only (skip installing the project package itself)
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application source code
COPY ./backend /app/backend

# Expose port for Cloud Run
EXPOSE 8000

# Start app with Cloud Run's PORT
CMD exec uvicorn --app-dir=backend main:app --host 0.0.0.0 --port $PORT
