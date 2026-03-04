from typing import Tuple


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg"
}


def validate_mime_type(mime_type: str) -> bool:
    return mime_type.lower() in ALLOWED_IMAGE_TYPES


def normalize_input(image_bytes: bytes, mime_type: str) -> Tuple[bytes, str]:
    if not image_bytes:
        raise ValueError("Image data is empty.")

    normalized_mime_type = (mime_type or "image/jpeg").strip().lower()
    if not validate_mime_type(normalized_mime_type):
        raise ValueError("Unsupported image format. Use JPG, PNG, or WEBP.")

    return image_bytes, normalized_mime_type
