import os
import json
from typing import Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiPredictor:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found.")
        
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model_name = 'gemini-2.5-flash' 

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.client:
            return {"prediction": "ERROR", "explanation": "API Key missing"}

        prompt = f"""
        Analyze the following news text for authenticity. 
        Determine if it is likely Real News or Fake News.
        
        Text:
        {text[:15000]}
        
        Respond with this JSON structure:
        {{
            "prediction": "REAL" or "FAKE",
            "confidence": <integer 0-100>,
            "explanation": "<short, clear explanation>"
        }}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json'
                )
            )
            
            raw_text = response.text.replace('```json', '').replace('```', '').strip()
            result = json.loads(raw_text)
            
            return {
                "prediction": result.get("prediction", "UNKNOWN").upper(),
                "confidence": result.get("confidence", 0),
                "explanation": result.get("explanation", "Analysis failed."),
                "meta": "Predicted Results"
            }
            
        except Exception as e:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "explanation": f"Error: {str(e)}",
                "meta": "System Error"
            }