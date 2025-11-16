# Dockerfile for Minecraft AI - MineRL Environment
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    openjdk-8-jdk \
    xvfb \
    x11vnc \
    fluxbox \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set JAVA_HOME for OpenJDK 8
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Create working directory
WORKDIR /app

# Copy environment files
COPY environment.yml requirements.txt* ./

# Install Python dependencies from requirements.txt if available
RUN if [ -f requirements.txt ]; then \
        pip3 install --no-cache-dir -r requirements.txt; \
    else \
        pip3 install --no-cache-dir \
            gym==0.23.1 \
            git+https://github.com/minerllabs/minerl \
            numpy==1.23.5 \
            opencv-python \
            torch torchvision \
            tensorboard \
            tqdm; \
    fi

# Copy project files
COPY . .

# Create directories for outputs
RUN mkdir -p videos runs logs

# Set default command
CMD ["/bin/bash"]

