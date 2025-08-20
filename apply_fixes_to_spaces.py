#!/usr/bin/env python3
"""
Script to apply Hugging Face Spaces fixes to medvisor-ai repository
"""

import os
import shutil
import subprocess
import sys

def run_command(command, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_git_repo(path):
    """Check if a path is a git repository"""
    return os.path.exists(os.path.join(path, '.git'))

def find_medvisor_ai_repo():
    """Find the medvisor-ai repository"""
    # Common locations to check
    possible_paths = [
        "../medvisor-ai",
        "../../medvisor-ai", 
        "./medvisor-ai",
        os.path.expanduser("~/medvisor-ai"),
        os.path.expanduser("~/Downloads/medvisor-ai"),
        os.path.expanduser("~/Desktop/medvisor-ai")
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and check_git_repo(path):
            print(f"Found medvisor-ai repository at: {path}")
            return path
    
    return None

def apply_fixes_to_repo(repo_path):
    """Apply all the fixes to the repository"""
    print(f"Applying fixes to: {repo_path}")
    
    # Files to copy (source -> destination)
    files_to_copy = [
        ("app.py", "app.py"),  # Updated Gradio-compatible app.py
        ("main.py", "main.py"),  # Enhanced entry point
        ("app_gradio.py", "app_gradio.py"),  # Gradio interface
        ("requirements_spaces.txt", "requirements.txt"),  # Updated requirements
        ("chat.py", "chat.py"),
        ("prediction.py", "prediction.py"),
        ("model.py", "model.py"),
        ("nltk_utils.py", "nltk_utils.py"),
        ("intents.json", "intents.json"),
        ("data.pth", "data.pth"),
    ]
    
    # Copy files
    for src, dst in files_to_copy:
        src_path = os.path.join(".", src)
        dst_path = os.path.join(repo_path, dst)
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                print(f"✓ Copied {src} -> {dst}")
            except Exception as e:
                print(f"✗ Failed to copy {src}: {e}")
        else:
            print(f"⚠ Source file not found: {src}")
    
    # Copy models directory if it exists
    models_src = "./models"
    models_dst = os.path.join(repo_path, "models")
    
    if os.path.exists(models_src):
        try:
            if os.path.exists(models_dst):
                shutil.rmtree(models_dst)
            shutil.copytree(models_src, models_dst)
            print("✓ Copied models directory")
        except Exception as e:
            print(f"✗ Failed to copy models directory: {e}")
    
    # Create README.md for the Space
    readme_content = """# MedVisor AI - Medical Image Analysis Platform

This Space provides AI-powered medical image analysis for spine conditions.

## Features

- **UNet Segmentation**: Automatic spine structure detection
- **Multi-class Classification**: Pfirrman grading and Modic changes  
- **Binary Classification**: Various spine pathologies
- **AI Chat Assistant**: Medical Q&A and guidance

## Usage

1. Upload a medical image (MRI, CT, X-ray)
2. Get automatic analysis results
3. Chat with the AI assistant for medical guidance

## Models

Models are automatically downloaded from `tudao01/spine` repository.

## Entry Point

This Space uses `main.py` as the entry point with Gradio interface.
"""
    
    readme_path = os.path.join(repo_path, "README.md")
    try:
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print("✓ Created README.md")
    except Exception as e:
        print(f"✗ Failed to create README.md: {e}")

def main():
    """Main function"""
    print("🔧 MedVisor AI - Hugging Face Spaces Fix Application")
    print("=" * 50)
    
    # Find the medvisor-ai repository
    repo_path = find_medvisor_ai_repo()
    
    if not repo_path:
        print("❌ Could not find medvisor-ai repository!")
        print("\nPlease provide the path to your medvisor-ai repository:")
        repo_path = input("Path: ").strip()
        
        if not os.path.exists(repo_path):
            print("❌ Invalid path!")
            return
        elif not check_git_repo(repo_path):
            print("❌ Not a git repository!")
            return
    
    # Apply fixes
    apply_fixes_to_repo(repo_path)
    
    print("\n✅ Fixes applied successfully!")
    print("\nNext steps:")
    print("1. Navigate to your repository:")
    print(f"   cd {repo_path}")
    print("2. Commit and push the changes:")
    print("   git add .")
    print("   git commit -m 'Fix: Use Gradio instead of Flask for Hugging Face Spaces'")
    print("   git push")
    print("3. Go to your Space settings and change:")
    print("   - Space SDK: Flask → Gradio")
    print("   - Entry point: app.py → main.py")
    print("4. Wait for the Space to restart and check the logs")

if __name__ == "__main__":
    main()
