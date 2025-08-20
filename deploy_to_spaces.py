#!/usr/bin/env python3
"""
Deployment script for Hugging Face Spaces
This script prepares your files for deployment to Hugging Face Spaces
"""

import os
import shutil
import subprocess
import sys

def create_spaces_directory(space_name="medvisor-ai"):
    """Create a directory structure for Hugging Face Spaces deployment"""
    
    # Create spaces directory
    spaces_dir = f"../{space_name}"
    if os.path.exists(spaces_dir):
        print(f"⚠️  Directory {spaces_dir} already exists. Removing...")
        shutil.rmtree(spaces_dir)
    
    os.makedirs(spaces_dir)
    print(f"✅ Created directory: {spaces_dir}")
    
    # Copy necessary files
    files_to_copy = [
        "main.py",
        "app_gradio.py", 
        "app.py",
        "chat.py",
        "prediction.py",
        "model.py",
        "nltk_utils.py",
        "intents.json",
        "data.pth"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, spaces_dir)
            print(f"✅ Copied: {file}")
        else:
            print(f"⚠️  Warning: {file} not found")
    
    # Copy requirements file
    if os.path.exists("requirements_spaces.txt"):
        shutil.copy2("requirements_spaces.txt", os.path.join(spaces_dir, "requirements.txt"))
        print("✅ Copied: requirements_spaces.txt -> requirements.txt")
    
    # Create README for the space
    readme_content = """# MedVisor AI - Medical Image Analysis

🏥 **MedVisor AI** is an advanced medical image analysis platform that combines deep learning models for spine condition analysis.

## Features

- **UNet Segmentation**: Automatic spine structure detection
- **Multi-class Classification**: Pfirrman grading and Modic changes  
- **Binary Classification**: Various spine pathologies
- **AI Chat Assistant**: Medical Q&A and guidance

## Usage

1. **Image Analysis**: Upload medical images for automated analysis
2. **Chat Assistant**: Ask questions about spine conditions and medical imaging

## Team

- Love Bhusal
- Tu Dao  
- Elden Delguia
- Riley Mckinney
- Sai Peram
- Rishil Uppaluru

**Note**: This tool is for educational and research purposes only. Always consult healthcare professionals for medical decisions.

---
*Powered by Hugging Face Spaces*
"""
    
    with open(os.path.join(spaces_dir, "README.md"), "w") as f:
        f.write(readme_content)
    print("✅ Created: README.md")
    
    return spaces_dir

def setup_git_repository(spaces_dir, space_url):
    """Initialize git repository and set up remote"""
    
    os.chdir(spaces_dir)
    
    # Initialize git repository
    subprocess.run(["git", "init"], check=True)
    print("✅ Initialized git repository")
    
    # Add remote
    subprocess.run(["git", "remote", "add", "origin", space_url], check=True)
    print(f"✅ Added remote: {space_url}")
    
    # Add all files
    subprocess.run(["git", "add", "."], check=True)
    print("✅ Added all files to git")
    
    # Initial commit
    subprocess.run(["git", "commit", "-m", "Initial MedVisor AI deployment"], check=True)
    print("✅ Created initial commit")
    
    print("\n🚀 Ready to deploy!")
    print(f"📁 Files prepared in: {spaces_dir}")
    print(f"🔗 Space URL: {space_url}")
    print("\nNext steps:")
    print("1. cd " + spaces_dir)
    print("2. git push -u origin main")
    print("3. Wait for deployment to complete")

def main():
    print("🚀 MedVisor AI - Hugging Face Spaces Deployment")
    print("=" * 50)
    
    # Get user input
    username = input("Enter your Hugging Face username: ").strip()
    space_name = input("Enter space name (default: medvisor-ai): ").strip() or "medvisor-ai"
    
    space_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    
    print(f"\n📋 Deployment Summary:")
    print(f"   Username: {username}")
    print(f"   Space Name: {space_name}")
    print(f"   Space URL: {space_url}")
    
    confirm = input("\nProceed with deployment? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Deployment cancelled")
        return
    
    try:
        # Create spaces directory
        spaces_dir = create_spaces_directory(space_name)
        
        # Setup git repository
        setup_git_repository(spaces_dir, space_url)
        
        print("\n🎉 Deployment setup complete!")
        print(f"📁 Your files are ready in: {spaces_dir}")
        
    except Exception as e:
        print(f"❌ Error during deployment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
