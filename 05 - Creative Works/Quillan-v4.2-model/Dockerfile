# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system deps if needed
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
