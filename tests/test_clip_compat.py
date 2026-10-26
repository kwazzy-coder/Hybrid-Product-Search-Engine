"""
CLIP must work with both transformers 4.x (get_*_features returns a tensor) and 5.x (returns
BaseModelOutputWithPooling). Uses a tiny randomly initialised CLIP, so nothing is downloaded.
Skipped when torch/transformers aren't installed.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))

from retrieval.clip_retriever import CLIPRetriever, embedding_tensor  # noqa: E402


@pytest.fixture(scope="module")
def tiny_clip():
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
                         vocab_size=99, max_position_embeddings=16, eos_token_id=2, bos_token_id=1, pad_token_id=0),
        vision_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
                           image_size=224, patch_size=32),
        projection_dim=512,
    )
    image_proc = CLIPImageProcessor()

    class Processor:
        def __call__(self, text=None, images=None, return_tensors="pt", **kw):
            if images is not None:
                return image_proc(images=images, return_tensors="pt")
            ids = torch.randint(3, 99, (len(text), 8))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    clip = CLIPRetriever()
    clip.model, clip.processor, clip._model_load_attempted = CLIPModel(cfg).eval(), Processor(), True
    return clip


def test_encode_and_search(tiny_clip):
    from PIL import Image

    images = [Image.new("RGB", (64, 64), (200, 30, 30)), Image.new("RGB", (64, 64), (30, 30, 200))]
    emb = tiny_clip.encode_images(images)
    assert emb.shape == (2, 512)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    tiny_clip.add_image_embeddings(emb, ["red", "blue"])
    assert tiny_clip.encode_texts(["red kurta"]).shape == (1, 512)
    assert tiny_clip.search_by_image(images[0])[0][0] == "red"
    assert len(tiny_clip.search("red kurta")) == 2


def test_embedding_tensor_accepts_every_output_shape():
    t = torch.ones(2, 512)

    class Output:
        pooler_output = t

    assert embedding_tensor(t) is t                      # transformers 4.x
    assert embedding_tensor(Output()) is t               # transformers 5.x
    assert embedding_tensor((torch.zeros(2, 3), t)) is t  # return_dict=False tuple
