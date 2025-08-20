#!/bin/bash

echo "Building optimized Docker image for MedVisor backend..."

# Build the image
docker build -t medvisor-backend:optimized .

echo "Build complete!"
echo ""
echo "To run the container:"
echo "docker run -p 5000:5000 medvisor-backend:optimized"
echo ""
echo "To check image size:"
echo "docker images medvisor-backend:optimized"
