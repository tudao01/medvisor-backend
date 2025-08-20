# Stage 1: Build
FROM python:3.12-slim AS build
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and pre-downloaded models
COPY . .          # Make sure models/ folder is included here

# Stage 2: Final
FROM python:3.12-slim
WORKDIR /app

# Copy installed packages from build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy app code and pre-downloaded models from build stage
COPY --from=build /app .

# Run your main script
CMD ["python", "main.py"]
