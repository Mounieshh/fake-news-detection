import base64
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


class ExternalImageApiBackend:
    # ext: {mime_type, media_type}
    SUPPORTED_FORMATS = {
        "png": ["image/png", "image_url"],
        "jpg": ["image/jpeg", "image_url"],
        "jpeg": ["image/jpeg", "image_url"],
        "webp": ["image/webp", "image_url"],
        "mp4": ["video/mp4", "video_url"],
        "webm": ["video/webm", "video_url"],
        "mov": ["video/mov", "video_url"]
    }

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = "nvidia/nemotron-nano-12b-v2-vl"
        self.query = "Describe the scene and identify any potential signs of fake or manipulated content"

    def infer(self, image_bytes: bytes, mime_type: str, context: str = "") -> Dict[str, Any]:
        if not self.api_key:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "explanation": "Image analysis model is not configured.",
                "meta": "System Error"
            }

        try:
            # Create temporary file from bytes
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            response_data = self._chat_with_media([tmp_path], self.query)
            os.unlink(tmp_path)

            return self._parse_response(response_data)

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

    def _get_extension(self, filename: str) -> str:
        """Extract file extension"""
        _, ext = os.path.splitext(filename)
        return ext[1:].lower()

    def _encode_media_base64(self, media_file: str) -> str:
        """Encode media file to base64 string"""
        with open(media_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _chat_with_media(self, media_files: List[str], query: str) -> Dict[str, Any]:
        """Call NVIDIA API with media files"""
        assert isinstance(media_files, list), f"media_files must be a list"

        has_video = False

        # Build content based on whether we have media files
        if len(media_files) == 0:
            content = query
        else:
            content = [{"type": "text", "text": query}]

            for media_file in media_files:
                ext = self._get_extension(media_file)
                assert ext in self.SUPPORTED_FORMATS, f"{media_file} format is not supported"

                media_type_key = self.SUPPORTED_FORMATS[ext][1]
                mime = self.SUPPORTED_FORMATS[ext][0]

                if media_type_key == "video_url":
                    has_video = True

                print(f"Encoding {media_file} as base64...", flush=True)
                base64_data = self._encode_media_base64(media_file)

                # Add media to content array
                media_obj = {
                    "type": media_type_key,
                    media_type_key: {
                        "url": f"data:{mime};base64,{base64_data}"
                    }
                }
                content.append(media_obj)

            if has_video:
                assert len(media_files) == 1, "Only single video supported."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        system_prompt = "/think"

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content,
            }
        ]

        payload = {
            "max_tokens": 4096,
            "temperature": 1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "messages": messages,
            "stream": False,
            "model": self.model,
        }

        response = requests.post(self.invoke_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse API response and extract prediction"""
        try:
            choices = response.get("choices", [])
            if not choices:
                return {
                    "prediction": "ERROR",
                    "confidence": 0,
                    "explanation": "No response from model.",
                    "meta": "System Error"
                }

            message_content = choices[0].get("message", {}).get("content", "")

            # Simple heuristic: check if content mentions fake/manipulated indicators
            lower_content = message_content.lower()
            fake_indicators = ["fake", "manipulated", "synthetic", "ai-generated", "deepfake", "edited", "altered"]
            real_indicators = ["authentic", "genuine", "original", "natural", "unedited"]

            fake_score = sum(1 for indicator in fake_indicators if indicator in lower_content)
            real_score = sum(1 for indicator in real_indicators if indicator in lower_content)

            if fake_score > real_score:
                prediction = "FAKE"
                confidence = min(100, (fake_score * 25))
            elif real_score > 0:
                prediction = "REAL"
                confidence = min(100, (real_score * 25))
            else:
                prediction = "UNCERTAIN"
                confidence = 50

            return {
                "prediction": prediction,
                "confidence": int(confidence),
                "explanation": message_content[:500],
                "meta": "Multimodal Vision Model Analysis"
            }

        except Exception as exc:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "explanation": f"Failed to parse API response: {str(exc)}",
                "meta": "System Error"
            }
