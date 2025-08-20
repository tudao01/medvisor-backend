# Stage 1: Build
FROM python:3.12-slim AS build
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Stage 2: Final
FROM python:3.12-slim
WORKDIR /app

# Copy installed packages and app code from build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /app .

CMD ["python", "main.py"]
