# Debugging Guide for Image Upload Issues

## Common Issues and Solutions

### 1. **Static Directory Issues**
**Problem**: Flask can't find static directories for file operations.
**Solution**: The app now automatically creates all necessary directories on startup.

### 2. **Model Loading Issues**
**Problem**: Models fail to load from Hugging Face Hub.
**Solution**: 
- Check if models exist in your Hugging Face repository
- Verify the repository ID: `tudao01/spine`
- Ensure models are named correctly:
  - `nonBinaryIndividualPredictions.keras`
  - `binaryIndividualPredictions.keras`
  - `unet_spine_segmentation.pth`

### 3. **File Upload Issues**
**Problem**: Frontend gets "image upload fail" error.
**Solutions**:
- Check CORS settings (currently set to allow all origins)
- Verify file size limits
- Check if the backend URL is correct in your frontend

### 4. **Testing the Backend**

#### Local Testing
```bash
# Start the backend
python app.py

# Test with the provided script
python test_upload.py http://localhost:5000 path/to/test/image.jpg
```

#### Railway Testing
```bash
# Test your deployed backend
python test_upload.py https://your-railway-app.railway.app path/to/test/image.jpg
```

### 5. **Common Error Messages**

#### "Models not loaded properly"
- Check Hugging Face repository access
- Verify model files exist
- Check logs for download errors

#### "Failed to process image with UNet"
- Check if OpenCV dependencies are installed
- Verify image format (should be JPG, PNG, etc.)
- Check if the image can be loaded

#### "Failed to split image"
- Check if the processed image has red contours
- Verify OpenCV is working properly
- Check image format and size

### 6. **Frontend Integration**

Make sure your frontend is sending the correct request:

```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('https://your-backend-url/upload', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.error) {
    console.error('Upload failed:', data.error);
  } else {
    console.log('Upload successful:', data);
  }
});
```

### 7. **Environment Variables**

For Railway deployment, make sure these are set:
- `PORT`: Railway will set this automatically
- `HOST`: Set to `0.0.0.0` for Railway

### 8. **Logs and Debugging**

The app now includes comprehensive logging. Check the logs for:
- Model download status
- File operation errors
- Image processing errors
- Prediction errors

### 9. **Size Limitations**

- **Railway Free Tier**: Limited to 512MB RAM and 1GB storage
- **Model Size**: Your models are ~7.9GB, which exceeds Railway's free tier
- **Solution**: Use Hugging Face Hub for model storage (already implemented)

### 10. **Performance Optimization**

- Models are loaded once at startup
- Images are processed in memory
- Static files are served efficiently
- Added verbose=0 to model predictions to reduce output

## Quick Fix Checklist

1. ✅ Static directories created automatically
2. ✅ Better error handling in upload route
3. ✅ Model loading with fallbacks
4. ✅ File path consistency fixed
5. ✅ OpenCV dependencies added to Dockerfile
6. ✅ Unique filenames for uploads
7. ✅ Comprehensive logging added

## Testing Steps

1. Deploy to Railway
2. Test health endpoint: `GET /health`
3. Test upload with a sample image
4. Check logs for any errors
5. Verify frontend can access the backend URL 