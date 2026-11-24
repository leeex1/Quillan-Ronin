FROM ubuntu:22.04

# Install dependencies including C++ build tools, ZeroMQ, PostgreSQL client libs, and PyZMQ for test client
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    libzmq3-dev \
    postgresql-client \
    libpq-dev \
    libpqxx-dev \
    python3 \
    python3-pip \
    curl \
    && pip3 install pyzmq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all application files
COPY . .

# Build the C++ MMORPG Server
RUN mkdir -p build && cd build && \
    cmake .. && \
    make && \
    cd ..
# Verify executable exists and set permissions
RUN ls -l /app/build/mmorpg_server
RUN chmod +x /app/build/mmorpg_server

# Expose ports for REQ/REP and PUB/SUB
EXPOSE 5555
EXPOSE 5556

# Command to run the MMORPG server
CMD ["/app/build/mmorpg_server"]
