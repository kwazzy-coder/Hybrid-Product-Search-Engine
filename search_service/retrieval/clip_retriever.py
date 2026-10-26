import os
import pickle
import threading
import faiss
import numpy as np

from paths import INDEX_DIR
from retrieval.ids import stable_id

def embedding_tensor(output):
    """
    CLIP's get_image_features / get_text_features return a plain tensor in transformers 4.x,
    but a BaseModelOutputWithPooling in 5.x, where the projected 512-d embedding is
    `pooler_output`. Accept both.
    """
    if hasattr(output, "norm"):          # already a torch.Tensor
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]                 # tuple form: (last_hidden_state, pooler_output, ...)
    raise TypeError(f"Unexpected CLIP output type: {type(output).__name__}")


class CLIPRetriever:
    def __init__(self):
        self.embedding_dim = 512
        self.model = None
        self.processor = None
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        self.id_map = {}
        self.exclusion_list = set()
        self._model_load_attempted = False
        self._load_lock = threading.Lock()
        self.load_error = None

    def _create_model(self):
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        model.eval()
        return model, processor

    def _load_model(self):
        """Load once. Concurrent callers wait for the load instead of seeing "no model"."""
        if self._model_load_attempted:
            return
        with self._load_lock:
            if self._model_load_attempted:
                return
            try:
                model, processor = self._create_model()
                self.processor = processor
                self.model = model
                print('CLIP model loaded for visual search.')
            except Exception as e:
                self.load_error = f"{type(e).__name__}: {e}"
                print(f'CLIP model unavailable: {self.load_error}. Visual search disabled.')
            finally:
                self._model_load_attempted = True

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        if self.model is None or self.processor is None:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        import torch
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            text_features = embedding_tensor(self.model.get_text_features(**inputs))
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().astype(np.float32)

    def encode_images(self, images) -> np.ndarray:
        self._load_model()
        if self.model is None or self.processor is None:
            return np.zeros((len(images), self.embedding_dim), dtype=np.float32)
        
        import torch
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            image_features = embedding_tensor(self.model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().astype(np.float32)

    def search_by_embedding(self, embedding: np.ndarray, top_k=100) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
            
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
            
        scores, ids = self.index.search(embedding.astype(np.float32), k)
        
        results = []
        for fid, score in zip(ids[0], scores[0]):
            fid_int = int(fid)
            if fid_int != -1 and fid_int in self.id_map:
                pid = self.id_map[fid_int]
                if pid in self.exclusion_list:
                    continue
                results.append((pid, float(score)))
                
        return results

    def search_by_image(self, image, top_k=100) -> list[tuple[str, float]]:
        self._load_model()
        if self.model is None or self.index.ntotal == 0:
            return []
            
        embedding = self.encode_images([image])
        return self.search_by_embedding(embedding, top_k=top_k)

    def search(self, query: str, top_k=100) -> list[tuple[str, float]]:
        self._load_model()
        if self.model is None:
            return []
            
        if self.index.ntotal == 0:
            return []
            
        embedding = self.encode_texts([query])
        return self.search_by_embedding(embedding, top_k=top_k)

    def add_image_embeddings(self, embeddings: np.ndarray, product_ids: list[str]):
        if len(product_ids) == 0:
            return
            
        incoming = set(product_ids)
        stale = [fid for fid, pid in self.id_map.items() if pid in incoming]
        if stale:
            self.index.remove_ids(np.array(stale, dtype=np.int64))
            for fid in stale:
                del self.id_map[fid]

        ids = []
        for pid in product_ids:
            fid = stable_id(pid)
            ids.append(fid)
            self.id_map[fid] = pid
            self.exclusion_list.discard(pid)
            
        ids_np = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(embeddings.astype("float32"), ids_np)
        print(f"CLIP index added {len(product_ids)} items. Total size: {self.index.ntotal}")

    def save(self, index_path=None, map_path=None):
        if index_path is None or map_path is None:
            if index_path is None:
                index_path = os.path.join(INDEX_DIR, "clip.index")
            if map_path is None:
                map_path = os.path.join(INDEX_DIR, "clip_id_map.pkl")
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        with open(map_path, "wb") as f:
            pickle.dump((self.id_map, self.exclusion_list), f)
        print(f"CLIP index and ID map saved to {index_path}")

    def load(self, index_path=None, map_path=None):
        if index_path is None or map_path is None:
            if index_path is None:
                index_path = os.path.join(INDEX_DIR, "clip.index")
            if map_path is None:
                map_path = os.path.join(INDEX_DIR, "clip_id_map.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(map_path):
            print(f"CLIP index or ID map files not found at {index_path} or {map_path}")
            return False
            
        self.index = faiss.read_index(index_path)
        with open(map_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                self.id_map, self.exclusion_list = data
            else:
                self.id_map = data
                self.exclusion_list = set()
        print(f"CLIP index loaded from {index_path}. Total size: {self.index.ntotal}")
        return True
