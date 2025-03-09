FROM python:3.10-slim

WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy requirements and install dependencies
COPY requirements.txt .
COPY pyproject.toml .
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY *.py .
COPY .env* .

# Create directories for data persistence
RUN mkdir -p Raw

# Expose the port the app runs on
EXPOSE 8188

# Command to run the application
CMD ["uvicorn", "pyAttender:app", "--host", "0.0.0.0", "--port", "8188"]