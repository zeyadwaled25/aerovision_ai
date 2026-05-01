# 🚀 AIC-4 Tracking Pipeline Dockerfile
# Base image with Python 3.10, PyTorch, and CUDA pre-installed
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# 🎯 Set the Entrypoint
# This allows organizers to run: docker run <image_name> --dataset_dir /data --split hidden_test
ENTRYPOINT ["python", "main.py"]