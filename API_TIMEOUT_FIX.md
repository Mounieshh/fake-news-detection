# API Timeout Fix and Troubleshooting Guide

## Problem: HTTPSConnectionPool Timeout

The error `Read timed out. (read timeout=60)` occurs when the NVIDIA API takes longer than expected to process your image.

## What Was Fixed

### 1. **Increased Timeout (60s → 300s)**
   - Changed from 60 seconds to 5 minutes (300 seconds)
   - Large images can take longer to analyze

### 2. **Added Retry Logic**
   - Automatically retries failed requests up to 3 times
   - Uses exponential backoff: 2s, 4s, 8s
   - Handles temporary API issues gracefully

### 3. **Connection Pooling**
   - Reuses HTTP connections for better performance
   - Reduces overhead on repeated requests

### 4. **Better Error Messages**
   - Distinguishes between different error types
   - Configuration errors (missing API key)
   - Connection errors (network issues)
   - API errors (rate limiting, invalid key)

### 5. **API Connection Test**
   - Added `test_connection()` method
   - Run `verify_setup.py` to test your API configuration

## How to Use

### Test Your API Configuration

Run the verification script:
```bash
python verify_setup.py
```

This will:
- ✓ Check if NVIDIA_API_KEY is configured
- ✓ Verify all required modules are installed
- ✓ Test the API connection
- ✓ Provide specific error messages if something fails

### Troubleshooting Steps

#### 1. **Missing or Invalid API Key**
```
Error: Invalid NVIDIA API key
```
**Solution:**
- Get your key from https://build.nvidia.com/
- Add to `.env` file:
  ```
  NVIDIA_API_KEY=your_api_key_here
  ```
- Don't use quotes around the key

#### 2. **API Rate Limiting**
```
Error: NVIDIA API rate limit exceeded
```
**Solution:**
- Wait a few moments before trying again
- Check your NVIDIA API quota
- The script will automatically retry

#### 3. **Network/Connection Issues**
```
Error: Cannot connect to API. Check internet connection.
```
**Solution:**
- Check your internet connection
- Try pinging api.nvidia.com
- Check if NVIDIA API is available at https://status.nvidia.com/

#### 4. **API is Slow/Overloaded**
```
Error: timeout after 300 seconds
```
**Solution:**
- The API service may be overloaded
- Try again in a few moments
- Try with smaller images first
- Check the image size (very large files take longer)

#### 5. **Image Size Issues**
```
Error during image encoding
```
**Solution:**
- Keep images under 20MB
- Common formats: PNG, JPG, WEBP
- Test with a smaller image first

## Configuration Files

### `.env` File
Create or update `.env` in the project root:
```
NVIDIA_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### `.env.example` File
Shows what variables are needed (for reference)

## Performance Tips

1. **Start Small**: Test with small images (< 1MB)
2. **Monitor**: Check console output for processing status
3. **Patience**: First request may take 30-60 seconds to process
4. **Batch Processing**: If processing multiple images, use the retry logic for reliability

## Code Changes Summary

**File:** `src/image_predict/_api_backend.py`

Changes made:
- ✓ Added session with retry strategy
- ✓ Increased timeout to 300 seconds
- ✓ Added connection pooling
- ✓ Better exception handling
- ✓ Added `test_connection()` method
- ✓ Improved error messages

**File:** `verify_setup.py`

Changes made:
- ✓ Added API connection testing
- ✓ Better error reporting

## Testing

### Quick Test
```bash
python verify_setup.py
```

### Full Test with Image
```python
from src.image_predict.image_model import ImageFakeNewsModel

model = ImageFakeNewsModel()
with open("test_image.jpg", "rb") as f:
    result = model.predict(f.read())
    print(result)
```

## Need Help?

1. Run `verify_setup.py` - it will show exactly what's wrong
2. Check your `.env` file has the correct API key
3. Look at the error message in the console - it now explains what went wrong
4. If API is timing out repeatedly, NVIDIA API may be experiencing issues

---

**Updated:** March 2026
