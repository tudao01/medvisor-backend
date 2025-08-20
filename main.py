"""
MedVisor AI - Medical Image Analysis Platform
Hugging Face Spaces Entry Point
"""

import os
import sys
import traceback

def main():
    """Main entry point for Hugging Face Spaces"""
    try:
        # Import and create the Gradio interface
        from app_gradio import create_interface
        
        print("🚀 Starting MedVisor AI on Hugging Face Spaces...")
        print("📥 Downloading models from Hugging Face Hub...")
        
        # Create and launch the interface
        demo = create_interface()
        
        print("✅ Models loaded successfully!")
        print("🌐 Launching Gradio interface...")
        
        # Launch with appropriate settings for Spaces
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Don't create public link in Spaces
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Available modules:")
        for module in sys.modules.keys():
            if "gradio" in module or "torch" in module or "tensorflow" in module:
                print(f"  - {module}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting MedVisor AI: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
