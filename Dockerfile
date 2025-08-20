# ------------------------------
# Stage 1: Build dependencies
# ------------------------------
FROM python:3.12-slim AS build
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and pre-downloaded models
COPY ./app ./app

# ------------------------------
# Stage 2: Final image
# ------------------------------
FROM python:3.12-slim
WORKDIR /app

# Copy installed Python packages from build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy app code and models from build stage
COPY --from=build /app ./app

# Set environment variables (optional)
ENV TF_ENABLE_ONEDNN_OPTS=0

# Set working directory to app
WORKDIR /app/app

# Expose port for Railway
EXPOSE 5000

# Run Flask app
CMD ["python", "main.py"]
