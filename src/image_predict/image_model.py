from typing import Any, Dict, Optional

from ._api_backend import ExternalImageApiBackend
from .preprocess import normalize_input


class ImageFakeNewsModel:
    def __init__(self, model_path: str = "model/image_model.pt"):
        self.model_path = model_path
        self.backend = None
        self.is_loaded = False
        self.error_message = None
        
        try:
            self.backend = ExternalImageApiBackend()
            self.is_loaded = True
        except ValueError as e:
            self.error_message = str(e)
            self.is_loaded = False

    def predict(self, image_bytes: bytes, mime_type: str, context: str = "") -> Dict[str, Any]:
        if not self.is_loaded:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "explanation": self.error_message or "Trained model failed to predict. Model is not initialized.",
                "meta": "Configuration Error"
            }
        
        try:
            image_bytes, mime_type = normalize_input(image_bytes=image_bytes, mime_type=mime_type)
        except ValueError as exc:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "explanation": str(exc),
                "meta": "Input Validation"
            }

        return self.backend.infer(image_bytes=image_bytes, mime_type=mime_type, context=context)


def load_model(model_path: str = "model/image_model.pt") -> ImageFakeNewsModel:
    return ImageFakeNewsModel(model_path=model_path)


def predict(
    model: ImageFakeNewsModel,
    image_bytes: bytes,
    mime_type: str,
    context: str = ""
) -> Dict[str, Any]:
    if not model or not model.is_loaded:
        error_msg = model.error_message if model else "Trained model is not initialized"
        return {
            "prediction": "ERROR",
            "confidence": 0,
            "explanation": error_msg or "Trained model failed to predict. Model is not loaded.",
            "meta": "Model Error"
        }

    return model.predict(image_bytes=image_bytes, mime_type=mime_type, context=context)
