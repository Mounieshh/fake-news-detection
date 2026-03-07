"""
Gemini-based Image Analysis Backend
Simple, fast, and reliable image fake news detection using Google's Gemini Vision API
"""

import os
from typing import Any, Dict
from io import BytesIO

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ExternalImageApiBackend:
    """Gemini Vision API backend for image analysis"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables. Please set it in .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"  # Gemini 2.5 Flash with vision support
    
    def infer(self, image_bytes: bytes, mime_type: str, context: str = "") -> Dict[str, Any]:
        """Analyze image for fake news/manipulation using Gemini Vision API"""
        
        try:
            print(f"Analyzing image with Gemini (size: {len(image_bytes) / 1024:.1f} KB)...", flush=True)
            
            # Prepare the analysis prompt
            prompt = """Analyze this image for signs of fake news, manipulation, or deepfakes.

Look for:
- AI-generated content or deepfakes
- Photo manipulation or editing artifacts  
- Misleading context or staging
- Inconsistencies in lighting, shadows, reflections
- Unnatural facial features or body proportions
- Signs of splicing or compositing

Respond ONLY with valid JSON in this exact format:
{
    "prediction": "REAL" or "FAKE",
    "confidence": <number between 0-100>,
    "explanation": "<brief explanation of your analysis>"
}"""
            
            # Upload the image
            upload_file = self.client.files.upload(
                file=BytesIO(image_bytes),
                config=types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name="image_analysis"
                )
            )
            
            print("Image uploaded, waiting for analysis...", flush=True)
            
            # Generate analysis with the image
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_uri(
                        file_uri=upload_file.uri,
                        mime_type=mime_type
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=0.3
                )
            )
            
            print("Analysis complete!", flush=True)
            return self._parse_response(response)

        except Exception as exc:
            print(f"Image analysis error: {str(exc)}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "explanation": f"Image analysis failed: {str(exc)}",
                "meta": "System Error"
            }
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse Gemini API response and extract prediction"""
        try:
            import json
            
            # Clean and parse JSON response
            raw_text = response.text.replace('```json', '').replace('```', '').strip()
            result = json.loads(raw_text)
            
            # Validate and normalize prediction
            prediction = result.get("prediction", "UNCERTAIN").upper()
            if prediction not in ["REAL", "FAKE"]:
                prediction = "UNCERTAIN"
            
            # Validate and clamp confidence
            confidence = int(result.get("confidence", 50))
            confidence = max(0, min(100, confidence))
            
            explanation = result.get("explanation", "Analysis completed")
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "explanation": explanation[:500],
                "meta": "Gemini Vision Analysis"
            }
            
        except Exception as exc:
            # Fallback: analyze raw text response
            print(f"JSON parsing failed, using text analysis: {str(exc)}", flush=True)
            try:
                text = response.text.lower()
                
                # Check for fake indicators
                fake_keywords = ["fake", "manipulated", "ai-generated", "deepfake", "edited", "synthetic"]
                real_keywords = ["real", "authentic", "genuine", "original", "unaltered"]
                
                fake_score = sum(1 for keyword in fake_keywords if keyword in text)
                real_score = sum(1 for keyword in real_keywords if keyword in text)
                
                if fake_score > real_score:
                    prediction = "FAKE"
                    confidence = min(75, 40 + fake_score * 15)
                elif real_score > 0:
                    prediction = "REAL"
                    confidence = min(75, 40 + real_score * 15)
                else:
                    prediction = "UNCERTAIN"
                    confidence = 50
                
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "explanation": response.text[:500],
                    "meta": "Gemini Vision Analysis"
                }
            except:
                return {
                    "prediction": "ERROR",
                    "confidence": 0,
                    "explanation": "Failed to parse API response",
                    "meta": "System Error"
                }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Gemini API connection"""
        if not self.api_key:
            return {
                "status": "ERROR",
                "message": "GEMINI_API_KEY not configured"
            }
        
        try:
            print("Testing Gemini API connection...", flush=True)
            # Quick test - generate simple content
            response = self.client.models.generate_content(
                model=self.model,
                contents="test"
            )
            
            if response.text:
                return {
                    "status": "SUCCESS",
                    "message": "Gemini API connection successful"
                }
            else:
                return {
                    "status": "ERROR",
                    "message": "API returned empty response"
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Gemini API connection failed: {str(e)}"
            }
