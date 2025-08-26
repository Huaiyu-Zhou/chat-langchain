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

# (Optional) If you want to install your package inside container, re-run poetry install without --no-root
# RUN poetry install --no-interaction --no-ansi

# Start app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]

