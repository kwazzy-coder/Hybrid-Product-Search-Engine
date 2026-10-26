"""
Embedding models load once, and searches that arrive while a model is still loading
wait for it (they used to get empty CLIP results, or random fallback vectors for MiniLM).
No torch needed: the model loaders are replaced with slow fakes.
"""

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))

from retrieval.clip_retriever import CLIPRetriever  # noqa: E402
from retrieval.faiss_retriever import FAISSRetriever  # noqa: E402


def run_concurrently(fn, n=3, stagger=0.1):
    results = [None] * n

    def worker(i):
        time.sleep(i * stagger)
        results[i] = fn()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


class SlowClip(CLIPRetriever):
    loads = 0

    def _create_model(self):
        SlowClip.loads += 1
        time.sleep(0.5)
        return object(), object()

    def encode_images(self, images):
        self._load_model()
        if self.model is None:
            return np.zeros((1, 512), dtype=np.float32)
        return np.full((1, 512), 1 / np.sqrt(512), dtype=np.float32)


def test_photo_searches_during_clip_load_wait_for_it():
    clip = SlowClip()
    clip.add_image_embeddings(np.full((1, 512), 1 / np.sqrt(512), dtype=np.float32), ["p1"])
    results = run_concurrently(lambda: clip.search_by_image("photo"))
    assert all(r and r[0][0] == "p1" for r in results)
    assert SlowClip.loads == 1                                  # loaded exactly once


class SlowMiniLM(FAISSRetriever):
    def _create_model(self):
        time.sleep(0.5)

        class Model:
            def encode(self, texts, normalize_embeddings=True):
                return np.full((len(texts), 384), 1 / np.sqrt(384), dtype=np.float32)
        return Model()


def test_text_searches_during_minilm_load_use_the_real_model():
    faiss = SlowMiniLM()
    real = np.full(384, 1 / np.sqrt(384), dtype=np.float32)
    vectors = run_concurrently(lambda: faiss._embed_texts(["linen shirt"])[0])
    assert all(np.allclose(v, real) for v in vectors)            # no random fallback vectors mid-load


def test_load_failure_is_reported():
    class Broken(CLIPRetriever):
        def _create_model(self):
            raise OSError("not enough memory")

    clip = Broken()
    clip.add_image_embeddings(np.full((1, 512), 1 / np.sqrt(512), dtype=np.float32), ["p1"])
    assert clip.search_by_image("photo") == []
    assert clip.load_error == "OSError: not enough memory"
