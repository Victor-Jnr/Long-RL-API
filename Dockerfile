FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install system dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        wget \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /workspace

# Expose application port
EXPOSE 53000

# To run the container with unlimited memory, port mapping, workspace mount, and a specific name:
# docker build -t longvila .
# docker run --gpus all -it --name LongRL -v /mnt/c/git/Long-RL-API:/workspace longvila:latest
# docker run --gpus all -it --name LongRL -p 53000:53000 -v /mnt/c/git/Long-RL-API:/workspace longvila:latest 