# Dockerfile for Minecraft AI - MineRL Environment
# Supports both CPU and GPU (CUDA) modes
ARG CUDA_VERSION=11.8

# Use CUDA base image for GPU support
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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
# Install PyTorch with CUDA support
RUN if [ -f requirements.txt ]; then \
        pip3 install --no-cache-dir -r requirements.txt; \
        # Reinstall PyTorch with CUDA if it was installed without CUDA
        pip3 install --no-cache-dir --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip3 install --no-cache-dir \
            gym==0.23.1 \
            git+https://github.com/minerllabs/minerl \
            numpy==1.23.5 \
            opencv-python \
            tensorboard \
            tqdm \
            torch torchvision --index-url https://download.pytorch.org/whl/cu118; \
    fi

# Copy project files
COPY . .

# Create directories for outputs
RUN mkdir -p videos runs logs

# Set default command
CMD ["/bin/bash"]

