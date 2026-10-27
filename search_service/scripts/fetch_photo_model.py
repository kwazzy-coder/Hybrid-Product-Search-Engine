"""
Download the photo-search model used in lite mode (CLIP ViT-B/32 image tower, 8-bit ONNX, ~88 MB)
into search_service/models_onnx/. Run it in your host's build step so cold starts don't have to
download it; without it, the API downloads the model the first time it starts.

    python search_service/scripts/fetch_photo_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.clip_onnx import DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL, download_model  # noqa: E402

if __name__ == "__main__":
    path = os.getenv("CLIP_ONNX_PATH") or DEFAULT_MODEL_PATH
    if os.path.exists(path):
        print(f"Photo model already present: {path}")
    else:
        download_model(path, os.getenv("CLIP_ONNX_URL") or DEFAULT_MODEL_URL)
