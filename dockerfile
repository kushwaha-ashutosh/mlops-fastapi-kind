# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire FastAPI app
COPY app/ ./app/

# Expose port for FastAPI
EXPOSE 8080

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
