"""
Photo search without PyTorch: CLIP ViT-B/32's image tower run with ONNX Runtime.

Used in SEARCH_MODE=lite, where torch doesn't fit (Render's free tier has 512 MB).
The 8-bit quantized vision model is ~88 MB on disk and ~200 MB in memory, so it fits
next to the BM25 + re-ranker engine. It's downloaded on first use (from Hugging Face
by default) and cached in search_service/models_onnx/.

Embeddings from this model are close to, but not identical with, the PyTorch CLIP
used in full mode, so it keeps its own image index (clip_onnx.index); the two are
never mixed.

Settings:
    CLIP_ONNX_PATH  where the model file lives (default search_service/models_onnx/vision_model_quantized.onnx)
    CLIP_ONNX_URL   where to download it from if it's missing
"""

from __future__ import annotations

import os
import threading

import numpy as np

from paths import INDEX_DIR
from retrieval.clip_retriever import CLIPRetriever

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(SERVICE_DIR, "models_onnx", "vision_model_quantized.onnx")
DEFAULT_MODEL_URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx"

# openai/clip-vit-base-patch32 preprocessing (its preprocessor_config.json)
IMAGE_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(image) -> np.ndarray:
    """PIL image -> (3, 224, 224) float32, matching CLIPImageProcessor:
    shortest side to 224 (bicubic), centre crop 224x224, scale to 0-1, normalise."""
    from PIL import Image

    image = image.convert("RGB")
    w, h = image.size
    # Like CLIPImageProcessor: short side exactly 224, long side int(long * 224 / short).
    if w <= h:
        new_w, new_h = IMAGE_SIZE, int(h * IMAGE_SIZE / w)
    else:
        new_w, new_h = int(w * IMAGE_SIZE / h), IMAGE_SIZE
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)


def download_model(path: str, url: str) -> None:
    import requests

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    print(f"Downloading the CLIP image model for photo search ({url}) ...")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    os.replace(tmp, path)
    print(f"  saved {os.path.getsize(path) / 1e6:.0f} MB to {path}")


class OnnxClipEncoder:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # One thread: small hosts have a fraction of a CPU, and this keeps search responsive.
        opts.intra_op_num_threads = int(os.getenv("CLIP_ONNX_THREADS", "1"))
        opts.inter_op_num_threads = 1
        # Memory arenas grow to the largest batch seen and never shrink; off keeps RSS low.
        opts.enable_cpu_mem_arena = False
        # Prepacking keeps a second, reordered copy of the weights: ~20 MB for a few % speed.
        opts.add_session_config_entry("session.disable_prepacking", "1")
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        names = [o.name for o in outputs]
        if "image_embeds" in names:
            self.output_name = "image_embeds"
        else:  # another export: take the (batch, 512) output, not the per-patch hidden states
            flat = [o.name for o in outputs if len(o.shape) == 2]
            self.output_name = flat[0] if flat else names[0]

    def encode(self, images) -> np.ndarray:
        batch = np.stack([preprocess(img) for img in images]).astype(np.float32)
        (emb,) = self.session.run([self.output_name], {self.input_name: batch})
        emb = emb.reshape(len(images), -1).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)


class OnnxCLIPRetriever(CLIPRetriever):
    """Same interface as CLIPRetriever (image side only), backed by ONNX Runtime."""

    def __init__(self, model_path: str | None = None, model_url: str | None = None):
        super().__init__()
        self.model_path = model_path or os.getenv("CLIP_ONNX_PATH") or DEFAULT_MODEL_PATH
        self.model_url = model_url or os.getenv("CLIP_ONNX_URL") or DEFAULT_MODEL_URL
        self._index_lock = threading.RLock()  # the index grows in the background while people search

    def _create_model(self):
        if not os.path.exists(self.model_path):
            download_model(self.model_path, self.model_url)
        return OnnxClipEncoder(self.model_path), "onnx"

    def encode_images(self, images) -> np.ndarray:
        self._load_model()
        if self.model is None:
            return np.zeros((len(images), self.embedding_dim), dtype=np.float32)
        return self.model.encode(images)

    def encode_texts(self, texts):
        # Text search in lite mode is BM25 only; the text tower isn't loaded.
        return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

    def search(self, query: str, top_k=100):
        return []

    def search_by_embedding(self, embedding, top_k=100):
        with self._index_lock:
            return super().search_by_embedding(embedding, top_k=top_k)

    def add_image_embeddings(self, embeddings, product_ids, verbose=True):
        with self._index_lock:
            super().add_image_embeddings(embeddings, product_ids, verbose=verbose)

    def save(self, index_path=None, map_path=None):
        with self._index_lock:
            super().save(index_path or os.path.join(INDEX_DIR, "clip_onnx.index"),
                         map_path or os.path.join(INDEX_DIR, "clip_onnx_id_map.pkl"))

    def load(self, index_path=None, map_path=None):
        with self._index_lock:
            return super().load(index_path or os.path.join(INDEX_DIR, "clip_onnx.index"),
                                map_path or os.path.join(INDEX_DIR, "clip_onnx_id_map.pkl"))
