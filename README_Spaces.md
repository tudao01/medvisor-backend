# MedVisor AI - Hugging Face Spaces Deployment

This repository contains the backend code for MedVisor AI, a medical image analysis platform that can be deployed on Hugging Face Spaces.

## 🚀 Quick Start

### 1. Create a New Space on Hugging Face

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose the following settings:
   - **Owner**: Your username
   - **Space name**: `medvisor-ai` (or your preferred name)
   - **Space SDK**: `Gradio`
   - **Space hardware**: `CPU` (or `GPU` if you have access)
   - **License**: Choose appropriate license

### 2. Upload Your Code

You can either:
- **Upload files directly** through the web interface
- **Use Git** to push your code to the space

#### Option A: Direct Upload
1. Upload all files from this directory to your Space
2. Make sure `main.py` is in the root directory
3. Ensure `requirements_spaces.txt` is renamed to `requirements.txt`

#### Option B: Git Push
```bash
# Clone your space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/medvisor-ai

# Copy your files
cp -r medvisor-backend/* medvisor-ai/

# Rename requirements file
mv medvisor-ai/requirements_spaces.txt medvisor-ai/requirements.txt

# Push to Hugging Face
cd medvisor-ai
git add .
git commit -m "Initial MedVisor AI deployment"
git push
```

### 3. Required Files Structure

Your Space should have this structure:
```
medvisor-ai/
├── main.py                    # Entry point for Spaces
├── app_gradio.py             # Gradio interface
├── app.py                    # Flask backend (fallback)
├── requirements.txt          # Python dependencies
├── chat.py                   # Chatbot functionality
├── prediction.py             # Image prediction functions
├── model.py                  # Neural network model
├── nltk_utils.py             # NLP utilities
├── intents.json              # Chatbot intents
├── data.pth                  # Trained chatbot model
└── README.md                 # Space description
```

### 4. Environment Variables (Optional)

You can set these in your Space settings:
- `HF_TOKEN`: Your Hugging Face token (if needed for private models)
- `MODEL_REPO_ID`: Override the default model repository

## 🔧 Configuration

### Model Repository
The app downloads models from `tudao01/spine` by default. You can change this in `app_gradio.py`:

```python
REPO_ID = "your-username/your-model-repo"
```

### Hardware Requirements
- **CPU**: Minimum 2GB RAM, recommended 4GB+
- **GPU**: T4 or better for faster inference
- **Storage**: At least 8GB for models and dependencies

## 📊 Features

### Image Analysis
- **UNet Segmentation**: Automatic spine structure detection
- **Multi-class Classification**: Pfirrman grading and Modic changes
- **Binary Classification**: Various spine pathologies
- **Real-time Processing**: Immediate results display

### AI Chat Assistant
- **Medical Q&A**: Answer questions about spine conditions
- **Context-aware**: Maintains conversation history
- **Educational**: Provides medical information and guidance

## 🛠️ Troubleshooting

### Common Issues

1. **Models not downloading**
   - Check your internet connection
   - Verify the model repository exists and is public
   - Check the logs for specific error messages

2. **Memory issues**
   - Switch to CPU-only deployment
   - Reduce batch size in model inference
   - Use smaller model variants

3. **Import errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check that file paths are correct
   - Verify Python version compatibility

### Logs and Debugging
- Check the Space logs in the Hugging Face interface
- Use the built-in error reporting in the Gradio interface
- Monitor resource usage in the Space settings

## 🔒 Security and Privacy

- **No data storage**: Images are processed in memory and not saved
- **Temporary files**: All temporary files are cleaned up after processing
- **Educational use**: This tool is for educational and research purposes only

## 📝 License

Make sure to include appropriate licensing information in your Space description and README.

## 🤝 Support

For issues specific to Hugging Face Spaces deployment:
1. Check the [Spaces documentation](https://huggingface.co/docs/hub/spaces)
2. Review the Space logs for error messages
3. Test locally first using the Flask version

## 🎯 Next Steps

After successful deployment:
1. Test all features thoroughly
2. Update your frontend to use the new Spaces URL
3. Monitor performance and resource usage
4. Consider adding more features or optimizations

---

**Note**: This deployment is optimized for Hugging Face Spaces. For production use, consider additional security measures and proper medical device regulations compliance.
