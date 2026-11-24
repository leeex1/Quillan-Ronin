FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Mock service for orchestrator
EXPOSE 3006

CMD ["python3", "-c", "\
import http.server; \
import socketserver; \
PORT = 3006; \
Handler = http.server.SimpleHTTPRequestHandler; \
with socketserver.TCPServer(('', PORT), Handler) as httpd: \
    print(f'orchestrator serving on port {PORT}'); \
    httpd.serve_forever()"]
