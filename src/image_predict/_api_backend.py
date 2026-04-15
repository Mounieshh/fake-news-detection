"""
Image Analysis Backend
Primary + silent fallback trained model backend for image-based fake news detection
"""

import os
import json
import base64
import requests
from typing import Any, Dict
from io import BytesIO

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Shared analysis prompt
def _build_prompt(context: str = "") -> str:
    context_block = f"Additional context: {context}" if context else ""
    return f"""You are an expert in digital forensics and media authentication. Analyze this image for signs of manipulation, AI generation, or misleading context.

Reason step-by-step through these dimensions before reaching a verdict:
1. Pixel-level artifacts: Are there compression artifacts, cloning patterns, or unnatural noise distributions?
2. Lighting and shadows: Are light sources consistent across all subjects and objects?
3. Facial feature anomalies: Are facial proportions, skin texture, or eye reflections natural?
4. Metadata signals: Does the visual content suggest post-processing or synthetic generation?
5. Contextual plausibility: Does the scene composition make physical and contextual sense?
{context_block}

After your step-by-step reasoning, produce a final verdict.

Important calibration rules:
- If visual evidence is ambiguous or insufficient, set confidence below 60.
- Confidence must be an integer between 0 and 100.

Respond ONLY with valid JSON:
{{
    "prediction": "REAL" or "FAKE",
    "confidence": <integer 0-100>,
    "explanation": "<concise explanation referencing your step-by-step analysis>"
}}"""


def _parse_json_result(raw_text: str) -> Dict[str, Any]:
    """Parse and normalize a JSON prediction response."""
    result = json.loads(raw_text.replace('```json', '').replace('```', '').strip())

    prediction = result.get("prediction", "UNCERTAIN").upper()
    if prediction not in ["REAL", "FAKE"]:
        prediction = "UNCERTAIN"

    confidence = max(0, min(100, int(result.get("confidence", 50))))
    if prediction in ("UNCERTAIN", "UNKNOWN"):
        confidence = min(confidence, 50)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "explanation": result.get("explanation", "Analysis completed")[:500],
        "meta": "Model Analysis"
    }


class ExternalImageApiBackend:
    """Trained model backend for image analysis with silent fallback"""

    def __init__(self):
        self.primary_key = os.getenv("GEMINI_API_KEY")
        self.fallback_key = os.getenv("NVIDIA_NIM_API_KEY")

        if not self.primary_key and not self.fallback_key:
            raise ValueError("Trained model is not configured. Please set the model key in .env file.")

        # Primary client (Gemini)
        self.primary_client = genai.Client(api_key=self.primary_key) if self.primary_key else None
        self.primary_model = "gemini-3-flash-preview"

        # Fallback (NIM - LLaMA 3.2 Vision)
        self.fallback_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.fallback_model = "meta/llama-3.2-90b-vision-instruct"

    def infer(self, image_bytes: bytes, mime_type: str, context: str = "") -> Dict[str, Any]:
        """Try primary model first, silently fall back to NIM if it fails."""

        # --- Primary: Gemini ---
        if self.primary_client:
            try:
                print(f"Analyzing image with primary model ({len(image_bytes)/1024:.1f} KB)...", flush=True)
                result = self._infer_primary(image_bytes, mime_type, context)
                if result.get("prediction") != "ERROR":
                    return result
                print("Primary model returned error, trying fallback...", flush=True)
            except Exception as exc:
                print(f"Primary model failed: {exc}, trying fallback...", flush=True)

        # --- Fallback: NIM LLaMA Vision ---
        if self.fallback_key:
            try:
                print("Analyzing image with fallback model...", flush=True)
                return self._infer_fallback(image_bytes, mime_type, context)
            except Exception as exc:
                print(f"Fallback model failed: {exc}", flush=True)

        return {
            "prediction": "ERROR",
            "confidence": 0,
            "explanation": "Trained model failed to predict. Please try again.",
            "meta": "Model Error"
        }

    def _infer_primary(self, image_bytes: bytes, mime_type: str, context: str) -> Dict[str, Any]:
        """Gemini Vision inference."""
        prompt = _build_prompt(context)

        upload_file = self.primary_client.files.upload(
            file=BytesIO(image_bytes),
            config=types.UploadFileConfig(mime_type=mime_type, display_name="image_analysis")
        )

        response = self.primary_client.models.generate_content(
            model=self.primary_model,
            contents=[
                types.Part.from_uri(file_uri=upload_file.uri, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                temperature=0.3
            )
        )

        return _parse_json_result(response.text)

    def _infer_fallback(self, image_bytes: bytes, mime_type: str, context: str) -> Dict[str, Any]:
        """NIM LLaMA 3.2 Vision inference."""
        prompt = _build_prompt(context)

        # Encode image as base64 data URL
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_image}"

        payload = {
            "model": self.fallback_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 1.00,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.fallback_key}",
            "Accept": "application/json"
        }

        response = requests.post(self.fallback_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        raw_text = response.json()["choices"][0]["message"]["content"]
        return _parse_json_result(raw_text)
