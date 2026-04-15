import os
import json
import logging
from typing import Dict, Any
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NIM_API_KEY = ""
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "meta/llama-3.3-70b-instruct"


def _build_text_prompt(text: str) -> str:
    return f"""You are a professional fact-checker and media literacy expert. Analyze the following news text for authenticity.

Reason step-by-step through these dimensions before reaching a verdict:
1. Source credibility signals: Does the text cite named sources, institutions, or verifiable references?
2. Writing style: Is the language sensationalist, emotionally charged, or uses excessive capitalization/punctuation?
3. Factual consistency: Are claims internally consistent? Do they contradict well-established facts?
4. Verifiable claims: Can the key assertions be checked against public records or reputable sources?
5. Opinion vs. fact: Is the content primarily opinion or clearly labeled satire?

After your step-by-step reasoning, produce a final verdict.

Important calibration rules:
- If the text contains insufficient verifiable signals, set confidence below 60.
- Do not produce overconfident predictions on ambiguous or opinion-based content.
- Confidence must be an integer between 0 and 100. Values above 85 are reserved for predictions supported by multiple strong, verifiable signals.

Text:
{text[:15000]}

Respond ONLY with valid JSON:
{{
    "prediction": "REAL" or "FAKE",
    "confidence": <integer 0-100>,
    "explanation": "<concise explanation referencing your step-by-step reasoning>"
}}"""


def _parse_result(raw: str) -> Dict[str, Any]:
    # Grounding returns free-form text — extract the JSON block from it
    import re
    # Try to find a JSON block (with or without ```json fences)
    json_match = re.search(r'\{[\s\S]*"prediction"[\s\S]*\}', raw)
    if json_match:
        raw = json_match.group(0)
    result = json.loads(raw.replace('```json', '').replace('```', '').strip())
    prediction = result.get("prediction", "UNKNOWN").upper()
    confidence = max(0, min(100, int(result.get("confidence", 0))))
    if prediction in ("UNCERTAIN", "UNKNOWN"):
        confidence = min(confidence, 50)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "explanation": result.get("explanation", "Analysis failed."),
        "meta": "Predicted Results"
    }


def _nim_fallback(text: str) -> Dict[str, Any]:
    """Silent fallback using NVIDIA NIM (Llama-3.3-70b) when Gemini is unavailable."""
    try:
        client = OpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY)
        completion = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "user", "content": _build_text_prompt(text)}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )
        raw = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                raw += chunk.choices[0].delta.content
        result = _parse_result(raw)
        result["meta"] = "Predicted Results"
        return result
    except Exception as e:
        logger.error(f"NIM fallback also failed: {e}")
        return {
            "prediction": "ERROR",
            "confidence": 0,
            "explanation": "Analysis is temporarily unavailable. Please try again shortly.",
            "meta": "Model Error"
        }


class NewsPredictor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("Model key not set. Please add GEMINI_API_KEY to your .env file.")

        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model = 'gemini-2.5-flash'

    def predict(self, text: str) -> Dict[str, Any]:
        # If Gemini is not configured, go straight to NIM fallback silently
        if not self.client:
            logger.info("Gemini not configured — using NIM fallback.")
            return _nim_fallback(text)

        try:
            # Use Google Search grounding for real-time web verification.
            # Note: grounding cannot be combined with response_mime_type='application/json',
            # so we parse JSON from the raw text response instead.
            response = self.client.models.generate_content(
                model=self.model,
                contents=_build_text_prompt(text),
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.3
                )
            )
            return _parse_result(response.text)
        except Exception as e:
            err = str(e)
            logger.warning(f"Gemini prediction failed ({err}) — falling back to NIM.")
            return _nim_fallback(text)
