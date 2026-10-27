"""
Photo search in lite mode (the 512 MB deployment): CLIP's image model on ONNX Runtime,
with the photo index built in the background after the API starts.

The real model is an 88 MB download, so the API tests swap in a tiny stand-in encoder
(an image's average colour) and check everything around it: the index filling up in
the background, what /api/stats reports along the way, and photo search results.
"""

import io
import os
import sys

import numpy as np
import pytest

from tests.test_api import CATALOG, _purge, load_app

COLOURS = {"libas_1": (150, 20, 40), "pl_1": (245, 245, 240), "tbh_1": (40, 70, 200), "pl_2": (15, 15, 15)}


class ColourEncoder:
    """Stand-in for OnnxClipEncoder: embeds an image as its mean colour."""

    def encode(self, images):
        out = np.zeros((len(images), 512), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, :3] = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0) / 255 + 0.05
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class ColourFetcher:
    def fetch(self, product):
        from PIL import Image
        return Image.new("RGB", (32, 32), COLOURS[product["_id"]])


def png(colour):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), colour).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def lite(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    main = load_app(tmp_path, monkeypatch, SEARCH_MODE="lite", PHOTO_INDEX_ON_START="0")
    main.clip._create_model = lambda: (ColourEncoder(), "test")
    main.photo_indexer.fetcher_factory = ColourFetcher
    with TestClient(main.app) as c:
        c.main = main
        yield c
    _purge()


def test_lite_photo_search_turns_on_once_indexed(lite):
    main = lite.main
    features = lite.get("/api/stats").json()["features"]
    assert features["search_mode"] == "lite" and features["visual_search_ready"] is False
    resp = lite.post("/api/search/image", files={"file": ("p.png", png((30, 60, 210)), "image/png")})
    assert resp.status_code == 503 and "Try again in a minute" in resp.json()["detail"]

    main.photo_indexer.run(CATALOG)  # what the background thread does at startup

    features = lite.get("/api/stats").json()["features"]
    assert features["visual_search_ready"] is True and features["visual_search_error"] is None
    assert features["photo_index"] == {**features["photo_index"], "state": "ready", "building": False,
                                       "indexed": 4, "total": 4, "failed": 0}
    body = lite.post("/api/search/image", files={"file": ("p.png", png((30, 60, 210)), "image/png")}).json()
    assert body["results"][0]["_id"] == "tbh_1" and body["results"][0]["_score_kind"] == "visual"
    assert "notice" not in body
    # Text still narrows a photo search.
    body = lite.post("/api/search/image", data={"q": "saree"},
                     files={"file": ("p.png", png((140, 25, 45)), "image/png")}).json()
    assert body["results"][0]["_id"] == "libas_1"
    # The index is saved, so a restart only embeds what's new.
    assert os.path.exists(os.path.join(main.INDEX_DIR, "clip_onnx.index"))


def test_partial_index_is_searchable_with_a_notice(lite):
    main = lite.main
    main.photo_indexer.run(CATALOG[:2])
    main.photo_indexer.state, main.photo_indexer.total = "indexing", 4
    stats = lite.get("/api/stats").json()["features"]
    assert stats["visual_search_ready"] is True and stats["photo_index"]["building"] is True
    body = lite.post("/api/search/image", files={"file": ("p.png", png((250, 250, 250)), "image/png")}).json()
    assert body["results"][0]["_id"] == "pl_1"
    assert "2 of 4 products" in body["notice"]


def test_model_download_failure_is_reported(lite):
    main = lite.main

    def offline():
        raise ConnectionError("huggingface.co unreachable")

    main.clip._create_model = offline
    main.photo_indexer.run(CATALOG)
    features = lite.get("/api/stats").json()["features"]
    assert features["visual_search_ready"] is False
    assert "huggingface.co unreachable" in features["visual_search_error"]


def test_indexer_interleaves_stores():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))
    from photo_indexer import interleave_by_store
    products = [{"_id": f"a{i}", "source": "a"} for i in range(3)] + [{"_id": "b0", "source": "b"}]
    assert [p["_id"] for p in interleave_by_store(products)] == ["a0", "b0", "a1", "a2"]


def test_image_fetcher_downloads_politely_and_respects_robots(tmp_path, monkeypatch):
    from tests.fake_sites import FakeSite, make_image_bytes
    monkeypatch.syspath_prepend(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))
    _purge()
    from photo_indexer import ImageFetcher

    shop = FakeSite()
    shop.routes["/robots.txt"] = (200, "text/plain", "User-agent: *\nDisallow: /private/\n")
    shop.routes["/cdn/shop/files/red.jpg"] = (200, "image/jpeg", make_image_bytes((200, 30, 30)))
    shop.routes["/private/blue.jpg"] = (200, "image/jpeg", make_image_bytes((30, 30, 200)))
    shop.start()
    try:
        fetcher = ImageFetcher()
        fetcher.images_dir = str(tmp_path)          # no saved thumbnails
        fetcher.session.min_delay = 0.0
        img = fetcher.fetch({"_id": "x1", "image_url": f"{shop.url}/cdn/shop/files/red.jpg?v=1"})
        assert img is not None and img.getpixel((5, 5))[0] > 150
        assert any("width=256" in hit for hit in shop.hits)   # asked the CDN for a small thumbnail
        assert fetcher.fetch({"_id": "x2", "image_url": f"{shop.url}/private/blue.jpg"}) is None
        assert not any(hit.startswith("/private/") for hit in shop.hits)
    finally:
        shop.stop()
        _purge()


def test_preprocessing_matches_clip_image_processor():
    transformers = pytest.importorskip("transformers")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))
    from PIL import Image
    from retrieval.clip_onnx import preprocess
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for h, w in [(400, 300), (300, 401), (97, 150)]]
    expected = transformers.CLIPImageProcessor()(images=images, return_tensors="np")["pixel_values"]
    assert np.abs(np.stack([preprocess(i) for i in images]) - expected).max() < 1e-5


def test_onnx_encoder_runs_a_model_file(tmp_path):
    """OnnxClipEncoder with a tiny ONNX graph (mean colour -> 512-d), in the real model's input/output names."""
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))
    from PIL import Image
    from retrieval.clip_onnx import OnnxClipEncoder

    weight = np.zeros((3, 512), dtype=np.float32)
    weight[0, 0] = weight[1, 1] = weight[2, 2] = 1.0
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
         helper.make_node("MatMul", ["pooled", "w"], ["image_embeds"]),
         helper.make_node("Identity", ["pooled"], ["last_hidden_state"])],
        "tiny_clip",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", 3]),
         helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["batch", 512])],
        [numpy_helper.from_array(weight, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))

    enc = OnnxClipEncoder(str(path))
    assert enc.output_name == "image_embeds"
    emb = enc.encode([Image.new("RGB", (300, 200), (255, 0, 0)), Image.new("RGB", (50, 80), (0, 0, 255))])
    assert emb.shape == (2, 512) and np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    assert emb[0].argmax() == 0 and emb[1].argmax() == 2
