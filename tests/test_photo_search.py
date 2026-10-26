"""
Photo search ranks by visual similarity (it used to return the same order for every photo,
because the text-trained re-ranker ignored the image score). Uses a tiny random CLIP, so
nothing is downloaded; skipped when torch/transformers aren't installed.
"""

import io
import random
import sqlite3
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.test_api import CATALOG, _purge, load_app  # noqa: E402


def patterned(seed: int):
    from PIL import Image
    rng = random.Random(seed)
    img = Image.new("RGB", (224, 224))
    px = img.load()
    for x in range(0, 224, 16):
        for y in range(0, 224, 16):
            colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for dx in range(16):
                for dy in range(16):
                    px[x + dx, y + dy] = colour
    return img


def png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def photo_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

    torch.manual_seed(0)
    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
                         vocab_size=99, max_position_embeddings=16, eos_token_id=2, bos_token_id=1, pad_token_id=0),
        vision_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
                           image_size=224, patch_size=32),
        projection_dim=512,
    )
    model, image_proc = CLIPModel(cfg).eval(), CLIPImageProcessor()

    class Processor:
        def __call__(self, text=None, images=None, return_tensors="pt", **kw):
            return image_proc(images=images, return_tensors="pt")

    main = load_app(tmp_path, monkeypatch)
    main.clip._create_model = lambda: (model, Processor())
    images = {p["_id"]: patterned(i) for i, p in enumerate(CATALOG)}
    main.clip._load_model()
    main.clip.add_image_embeddings(main.clip.encode_images(list(images.values())), list(images))
    with TestClient(main.app) as c:
        c.images, c.tmp_path = images, tmp_path
        yield c
    _purge()


def test_each_products_own_photo_comes_first(photo_client):
    for pid, img in photo_client.images.items():
        body = photo_client.post("/api/search/image", files={"file": ("p.png", png(img), "image/png")}).json()
        first = body["results"][0]
        assert first["_id"] == pid, f"uploaded {pid}'s photo, got {first['_id']} first"
        assert first["_score_kind"] == "visual" and first["_similarity"] > 0.99
        sims = [r["_similarity"] for r in body["results"]]
        assert sims == sorted(sims, reverse=True)                 # most similar first


def test_text_with_a_photo_filters(photo_client):
    img = photo_client.images["libas_1"]                          # a ₹2,499 saree
    body = photo_client.post("/api/search/image", data={"q": "under 1000"},
                             files={"file": ("p.png", png(img), "image/png")}).json()
    assert body["results"] and all(r["price"] <= 1000 for r in body["results"])
    assert "libas_1" not in [r["_id"] for r in body["results"]]


def test_phone_photo_rotation_and_bad_files(photo_client):
    from PIL import Image
    img = photo_client.images["pl_2"]
    # A phone photo stored sideways with an EXIF "rotate 90°" tag should match the upright original.
    sideways = img.transpose(Image.Transpose.ROTATE_90)          # stored turned 90° anticlockwise...
    exif = Image.Exif()
    exif[0x0112] = 6                                               # ...with the tag that says "rotate 90° clockwise to view"
    buf = io.BytesIO()
    sideways.save(buf, "PNG", exif=exif.tobytes())
    body = photo_client.post("/api/search/image", files={"file": ("phone.png", buf.getvalue(), "image/png")}).json()
    assert body["results"][0]["_id"] == "pl_2"
    bad = photo_client.post("/api/search/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 400 and "JPG or PNG" in bad.json()["detail"]


def test_photo_searches_are_not_used_for_retraining(photo_client):
    sys.modules["main"]  # app is loaded
    body = photo_client.post("/api/search/image",
                             files={"file": ("p.png", png(photo_client.images["pl_1"]), "image/png")}).json()
    clicked = body["results"][1]
    photo_client.post("/api/feedback", json={"query": "[photo]", "query_id": body["query_id"],
                                             "clicked_id": clicked["_id"], "position": 1})
    from feedback_db import get_training_sessions
    assert get_training_sessions(1) == []
    db = sqlite3.connect(photo_client.tmp_path / "feedback.db")
    assert db.execute("SELECT COUNT(*) FROM impressions WHERE mode='image'").fetchone()[0] > 0
    db.close()
