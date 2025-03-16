FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY .env* ./

# Create directories for data persistence
RUN mkdir -p Raw

# Expose the port the app runs on
EXPOSE 8188

# Command to run the application
CMD ["uvicorn", "zoom_attendance:app", "--host", "0.0.0.0", "--port", "8188"]