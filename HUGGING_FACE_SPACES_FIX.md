# 🔧 Hugging Face Spaces Fix Guide

## 🚨 The Problem

You're getting this error on Hugging Face Spaces:
```
ModuleNotFoundError: No module named 'flask'
```

This happens because:
1. **Wrong Entry Point**: Hugging Face Spaces is trying to run `app.py` (Flask) instead of the correct Gradio entry point
2. **Missing Dependencies**: Flask dependencies aren't being installed properly
3. **Incorrect Space Configuration**: The Space should be configured for Gradio, not Flask

## ✅ The Solution

### Step 1: Fix Your Space Configuration

1. **Go to your Hugging Face Space settings**
2. **Change the Space SDK to "Gradio"** (not Flask)
3. **Set the entry point to `main.py`** (not `app.py`)

### Step 2: Update Your Files

I've already fixed your files. Here's what changed:

#### ✅ `app.py` - Now Gradio-compatible
- Changed from Flask to Gradio entry point
- Redirects to `app_gradio.py` for the interface
- Includes proper dependency checking

#### ✅ `requirements_spaces.txt` - Updated dependencies
```txt
gradio>=4.0.0
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.10.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
segmentation-models-pytorch>=0.3.0
huggingface-hub>=0.16.0
nltk>=3.8.0
matplotlib>=3.7.0
pandas>=2.0.0
flask>=2.0.0
flask-cors>=3.0.0
```

#### ✅ `main.py` - Enhanced entry point
- Better error handling and dependency checking
- Clear startup messages
- Proper Gradio configuration

#### ✅ `app_flask.py` - Flask backup
- Original Flask application preserved
- Use for local development or other deployments

### Step 3: Deploy to Hugging Face Spaces

#### Option A: Manual Upload
1. **Go to your Space on Hugging Face**
2. **Upload these files**:
   - `main.py` (entry point)
   - `app_gradio.py` (Gradio interface)
   - `requirements_spaces.txt` → rename to `requirements.txt`
   - All other Python files (`chat.py`, `prediction.py`, etc.)
   - Model files and data

#### Option B: Use Git
```bash
# Clone your space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/medvisor-ai

# Copy the fixed files
cp main.py medvisor-ai/
cp app_gradio.py medvisor-ai/
cp requirements_spaces.txt medvisor-ai/requirements.txt
cp *.py medvisor-ai/
cp *.json medvisor-ai/
cp *.pth medvisor-ai/

# Push to Hugging Face
cd medvisor-ai
git add .
git commit -m "Fix: Use Gradio instead of Flask for Hugging Face Spaces"
git push
```

### Step 4: Verify Your Space Settings

In your Space settings, ensure:
- **Space SDK**: `Gradio`
- **Python version**: `3.9` or `3.10`
- **Hardware**: `CPU` (or `GPU` if available)
- **Entry point**: `main.py`

## 🔍 Troubleshooting

### If you still get errors:

#### 1. Check Space Logs
- Go to your Space → "Logs" tab
- Look for specific error messages
- Check if models are downloading

#### 2. Test Dependencies
The new `main.py` will check dependencies and show:
```
✓ Gradio imported successfully
✓ PyTorch imported successfully
✓ TensorFlow imported successfully
```

#### 3. Model Download Issues
If models fail to download:
- Check if `tudao01/spine` repository is public
- Verify the model filenames match exactly
- Check your internet connection in the Space

#### 4. Memory Issues
If you get memory errors:
- Switch to CPU-only deployment
- Reduce model batch sizes
- Use smaller model variants

## 🎯 Expected Behavior

After fixing, your Space should:

1. **Start successfully** with dependency checks
2. **Download models** from Hugging Face Hub
3. **Launch Gradio interface** on port 7860
4. **Show the MedVisor AI interface** with:
   - Image upload functionality
   - Chat interface
   - Model predictions

## 📞 Quick Commands

### Test Locally (Flask)
```bash
python app_flask.py
```

### Test Locally (Gradio)
```bash
python main.py
```

### Deploy to Spaces
```bash
# Use the deployment script
python deploy_to_spaces.py
```

## 🎉 Success Indicators

You'll know it's working when you see:
- ✅ "Starting MedVisor AI on Hugging Face Spaces..."
- ✅ "✓ Gradio imported successfully"
- ✅ "All models downloaded successfully!"
- ✅ "Launching Gradio interface..."
- ✅ Your Space shows the MedVisor AI interface

## 🔄 Alternative: Use Flask (Not Recommended for Spaces)

If you really need Flask on Hugging Face Spaces:

1. **Change Space SDK to "Docker"**
2. **Use the original `app.py`** (I've saved it as `app_flask.py`)
3. **Create a proper `Dockerfile`**
4. **Set entry point to `app.py`**

But **Gradio is strongly recommended** for Hugging Face Spaces as it's:
- ✅ Better integrated
- ✅ Easier to deploy
- ✅ More reliable
- ✅ Better for ML applications

---

**🎯 The key fix**: Change your Space SDK from Flask to Gradio and use `main.py` as the entry point!
