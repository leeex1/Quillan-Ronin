# 1. Start from a suitable base image
FROM ubuntu:22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install essential C++ development tools and JUCE dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    cmake \
    make \
    git \
    libasound2-dev \
    libjack-jackd2-dev \
    ladspa-sdk \
    libcurl4-openssl-dev \
    libx11-dev \
    libxinerama-dev \
    libxext-dev \
    libxrandr-dev \
    libxcursor-dev \
    libgl1-mesa-dev \
    libwebkit2gtk-4.0-dev || \
    apt-get install -y libwebkit2gtk-4.1-dev

# 3. Clone JUCE library
RUN git clone https://github.com/juce-framework/JUCE.git /opt/JUCE
# Verify JUCE clone
RUN ls -la /opt/JUCE
# Example: Checkout a specific version/tag if needed
# RUN cd /opt/JUCE && git checkout 7.0.0

# 4. Clone Tracktion Engine library
RUN git clone https://github.com/Tracktion/tracktion_engine.git /opt/tracktion_engine
# Verify Tracktion Engine clone
RUN ls -la /opt/tracktion_engine

# 5. Set up environment variables
ENV JUCE_PATH=/opt/JUCE

# 6. Set WORKDIR
WORKDIR /app/daw_project

# 7. Add a placeholder CMD
CMD ["tail", "-f", "/dev/null"]
