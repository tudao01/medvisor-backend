# Multi-stage build for optimized Python backend
FROM python:3.9-alpine

# Install system dependencies for building
RUN apk add --no-cache gcc g++ musl-dev libffi-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements-optimized.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-optimized.txt

# Runtime stage
FROM python:3.9-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only necessary Python files
COPY app.py .
COPY model.py .
COPY prediction.py .
COPY chat.py .
COPY nltk_utils.py .
COPY intents.json .
COPY data.pth .

# Create output directory
RUN mkdir -p static/output

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]

