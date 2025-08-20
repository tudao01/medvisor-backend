# 🚀 MedVisor AI Deployment Guide

This guide will help you deploy MedVisor AI to Hugging Face Spaces and update your frontend to work with the new backend.

## 📋 Prerequisites

- Hugging Face account
- Git installed on your machine
- Your models uploaded to Hugging Face Hub (already done: `tudao01/spine`)

## 🎯 Step 1: Deploy to Hugging Face Spaces

### Option A: Automated Deployment (Recommended)

1. **Run the deployment script:**
   ```bash
   cd medvisor-backend
   python deploy_to_spaces.py
   ```

2. **Follow the prompts:**
   - Enter your Hugging Face username
   - Choose a space name (default: `medvisor-ai`)
   - Confirm the deployment

3. **Push to Hugging Face:**
   ```bash
   cd ../medvisor-ai
   git push -u origin main
   ```

### Option B: Manual Deployment

1. **Create a new Space:**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose:
     - **Owner**: Your username
     - **Space name**: `medvisor-ai`
     - **Space SDK**: `Gradio`
     - **Space hardware**: `CPU` (or `GPU` if available)
     - **License**: Choose appropriate license

2. **Upload files manually:**
   - Upload all files from `medvisor-backend/` to your Space
   - Rename `requirements_spaces.txt` to `requirements.txt`
   - Ensure `main.py` is in the root directory

## 🔧 Step 2: Configure Your Space

### Environment Variables (Optional)
In your Space settings, you can add:
- `HF_TOKEN`: Your Hugging Face token (if needed)
- `MODEL_REPO_ID`: Override default model repository

### Hardware Settings
- **CPU**: Minimum 2GB RAM, recommended 4GB+
- **GPU**: T4 or better for faster inference
- **Storage**: At least 8GB for models and dependencies

## 🌐 Step 3: Update Your Frontend

### Update Environment Variables

1. **Create/update `.env` file in your frontend:**
   ```bash
   cd medvisor-deployment
   ```

2. **Add your Spaces URL:**
   ```env
   REACT_APP_SPACES_URL=https://your-username-medvisor-ai.hf.space
   ```

### Update API Calls

The frontend has been updated to use the new API utilities. Key changes:

1. **Image Upload**: Now uses Gradio's API format
2. **Chat**: Uses the new chat endpoint
3. **Health Check**: Monitors Space status

### Test the Integration

1. **Start your frontend:**
   ```bash
   npm start
   ```

2. **Test image upload:**
   - Upload a medical image
   - Check if processing works
   - Verify results display correctly

3. **Test chat functionality:**
   - Open the chat interface
   - Send a test message
   - Verify responses

## 🔍 Step 4: Troubleshooting

### Common Issues

#### 1. Models Not Downloading
**Symptoms:** Space shows "Models not loaded" error
**Solutions:**
- Check if your model repository is public
- Verify the repository ID in `app_gradio.py`
- Check Space logs for specific errors

#### 2. Memory Issues
**Symptoms:** Space crashes or times out
**Solutions:**
- Switch to CPU-only deployment
- Reduce model batch sizes
- Use smaller model variants

#### 3. Frontend Connection Issues
**Symptoms:** "Failed to fetch" errors
**Solutions:**
- Verify the Spaces URL is correct
- Check if the Space is running
- Ensure CORS is properly configured

### Debugging Steps

1. **Check Space Logs:**
   - Go to your Space on Hugging Face
   - Click "Logs" tab
   - Look for error messages

2. **Test Space Directly:**
   - Visit your Space URL directly
   - Try uploading an image
   - Test the chat function

3. **Verify Model Repository:**
   - Visit `https://huggingface.co/tudao01/spine`
   - Ensure all models are available
   - Check file permissions

## 📊 Step 5: Monitor and Optimize

### Performance Monitoring
- Monitor Space resource usage
- Check response times
- Track error rates

### Optimization Tips
- Use GPU if available for faster inference
- Implement caching for frequently used models
- Consider model quantization for smaller sizes

## 🔒 Step 6: Security Considerations

### Data Privacy
- Images are processed in memory only
- No data is stored permanently
- Temporary files are cleaned up

### Access Control
- Consider making your Space private if needed
- Implement rate limiting if required
- Monitor usage patterns

## 🎉 Step 7: Go Live!

Once everything is working:

1. **Deploy your frontend to Vercel:**
   ```bash
   cd medvisor-deployment
   vercel --prod
   ```

2. **Update your domain settings:**
   - Point your custom domain to Vercel
   - Update DNS records if needed

3. **Test the complete system:**
   - Test from different devices
   - Verify all features work
   - Check performance

## 📞 Support

### Getting Help
1. Check the [Hugging Face Spaces documentation](https://huggingface.co/docs/hub/spaces)
2. Review Space logs for specific errors
3. Test components individually

### Useful Links
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Vercel Deployment](https://vercel.com/docs)

---

## 🎯 Quick Reference

### Space URL Format
```
https://your-username-medvisor-ai.hf.space
```

### Key Files
- `main.py` - Entry point for Spaces
- `app_gradio.py` - Gradio interface
- `requirements.txt` - Dependencies
- `README.md` - Space description

### Environment Variables
```env
REACT_APP_SPACES_URL=https://your-username-medvisor-ai.hf.space
```

### Commands
```bash
# Deploy to Spaces
python deploy_to_spaces.py

# Push to Hugging Face
git push -u origin main

# Deploy frontend to Vercel
vercel --prod
```

---

**🎉 Congratulations!** Your MedVisor AI is now deployed on Hugging Face Spaces with a modern, scalable architecture that handles your large models efficiently.
