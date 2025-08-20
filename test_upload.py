#!/usr/bin/env python3
"""
Test script to debug upload functionality
"""
import requests
import os
import sys

def test_health_check(base_url):
    """Test if the server is running"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_upload(base_url, image_path):
    """Test image upload"""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            print(f"Uploading {image_path} to {base_url}/upload")
            response = requests.post(f"{base_url}/upload", files=files)
        
        print(f"Upload status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("Upload successful!")
            print(f"Output image URL: {result.get('output_image_url')}")
            print(f"Number of disc images: {len(result.get('disc_images', []))}")
            return True
        else:
            print("Upload failed!")
            return False
            
    except Exception as e:
        print(f"Upload test failed: {e}")
        return False

def main():
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "test_image.jpg"
    
    print(f"Testing backend at: {base_url}")
    print(f"Using test image: {image_path}")
    print("-" * 50)
    
    # Test health check
    if not test_health_check(base_url):
        print("Server is not responding. Please check if the backend is running.")
        return
    
    print("-" * 50)
    
    # Test upload
    test_upload(base_url, image_path)

if __name__ == "__main__":
    main() 