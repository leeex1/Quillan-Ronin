FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Expose the port the actual AI Assistant application (main.py) runs on
EXPOSE 5000

# Command to run the actual AI Assistant application
# Assumes main.py is in the WORKDIR (/app)
CMD ["python3", "main.py"]
