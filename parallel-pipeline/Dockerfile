FROM intel/dlstreamer:2025.2.0-ubuntu22

USER root

# Build args for proxy (pass via --build-arg)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

# Set proxy env vars
ENV HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY \
    NO_PROXY=$NO_PROXY \
    http_proxy=$http_proxy \
    https_proxy=$https_proxy \
    no_proxy=$no_proxy

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        git \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        python3-gi \
        python3-gst-1.0 \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from parallel-pipeline directory
COPY parallel-pipeline/requirements.txt .

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    opencv-python-headless \
    ultralytics \
    minio \
    easyocr \
    openvino \
    openvino-genai \
    torch \
    torchvision

# Install requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy parallel-pipeline code
COPY parallel-pipeline/ .

# Copy application-service and frame-selector-service
COPY application-service/app/ /app/application-service/app/
COPY frame-selector-service/app/ /app/frame-selector-service/app/

# Set Python path to include /app first (for local frame_pipeline.py), then sibling directories and DL Streamer (CRITICAL for gvapython)
ENV PYTHONPATH=/app:/app/application-service/app:/app/frame-selector-service/app:/opt/intel/dlstreamer/python:/opt/intel/dlstreamer/gstreamer/lib/python3/dist-packages:/home/dlstreamer/dlstreamer/python

# Set GStreamer plugin path
ENV GST_PLUGIN_PATH=/opt/intel/dlstreamer/lib:/opt/intel/dlstreamer/gstreamer/lib/gstreamer-1.0:/opt/intel/dlstreamer/gstreamer/lib/
ENV GST_VA_ALL_DRIVERS=1

# Default command
CMD ["python3", "main.py", "fixed", "--stations", "2"]
