# ------------------------------
# Stage 1: Build dependencies
# ------------------------------
FROM python:3.12-slim AS build
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code and models
COPY . .
COPY models/ ./models/

# ------------------------------
# Stage 2: Final image
# ------------------------------
FROM python:3.12-slim
WORKDIR /app

# Install runtime dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy code and models from build stage
COPY --from=build /app .

# Create necessary directories
RUN mkdir -p static/uploads static/output/discs models

# Set environment variables
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1

# Expose port for Railway
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
