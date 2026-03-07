# API Configuration Guide

This project uses two APIs for fake news detection:

## 1. Image Analysis API (NVIDIA Nemotron Vision)

### Get Your NVIDIA API Key:
1. Visit https://build.nvidia.com/
2. Sign up or log in to your account
3. Navigate to "API keys" section
4. Generate a new API key
5. Copy and paste it to your `.env` file as `NVIDIA_API_KEY`

### Example:
```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 2. Text Analysis API (Google Gemini)

### Get Your Gemini API Key:
1. Visit https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key or use existing one
4. Copy and paste it to your `.env` file as `GEMINI_API_KEY`

### Example:
```
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Setup Instructions

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to `.env`:**
   ```bash
   # Edit .env and replace placeholder values with your actual API keys
   NVIDIA_API_KEY=your_actual_nvidia_key_here
   GEMINI_API_KEY=your_actual_gemini_key_here
   ```

3. **Start the application:**
   ```bash
   python main.py
   ```

4. **Test image prediction:**
   - Upload an image through the web interface at `http://localhost:5000`
   - The API will analyze the image for signs of manipulation or deepfakes

---

## Troubleshooting

### Error: "NVIDIA_API_KEY not set in environment variables"
- Make sure your `.env` file exists in the project root directory
- Verify that `NVIDIA_API_KEY` is set to a valid API key
- Restart the application after updating `.env`

### Error: "GEMINI_API_KEY not found"
- Make sure your `.env` file contains the `GEMINI_API_KEY`
- Verify the key is valid and not expired
- Restart the application

### Image Analysis Not Working
- Ensure the image format is supported (JPG, PNG, WEBP)
- Check that your NVIDIA API quota hasn't been exceeded
- Verify internet connection for API calls

