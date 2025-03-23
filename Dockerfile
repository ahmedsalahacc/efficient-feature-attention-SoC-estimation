# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install notebook
# Copy project files (if needed)
COPY . .

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ensure NVIDIA runtime is used
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

EXPOSE 8888

# Define entrypoint
# CMD ["python", "src/main.py"]
CMD ["jupyter-notebook","--allow-root"]
