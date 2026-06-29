import os
import pickle
import faiss
import numpy as np
import hashlib

# Lazy import SentenceTransformer; provide fallback if it's unavailable

class FAISSRetriever:
    def __init__(self):
        # Preferred embedding dimension (matches all-MiniLM-L6-v2)
        self.embedding_dim = 384
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("Loaded SentenceTransformer model for FAISS retriever.")
        except Exception as e:
            print(f"SentenceTransformer unavailable: {e}. Using deterministic fallback embeddings.")

        # Inner Product similarity (normalize first = cosine similarity)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        self.id_map = {}           # faiss int64 id → product_id string
        self.exclusion_list = set() # product_ids to exclude (deleted products)

    def _fallback_embed(self, text: str) -> np.ndarray:
        # Deterministic pseudo-embedding based on SHA256-derived seed
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim).astype("float32")
        # Normalize to unit vector for cosine-sim using inner product
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True)
        # Fallback deterministic embeddings
        emb_list = [self._fallback_embed(t if t is not None else "") for t in texts]
        return np.vstack(emb_list)

    def add_products(self, products: list[dict]):
        if not products:
            return
        texts = [p.get("embedding_text", "") for p in products]
        embeddings = self._embed_texts(texts)
        
        ids = []
        for p in products:
            # Generate stable 63-bit integer hash to avoid sign overflow in FAISS
            fid = int(hash(p["_id"]) & 0x7FFFFFFFFFFFFFFF)
            ids.append(fid)
            self.id_map[fid] = p["_id"]
            # Remove from exclusion list if it is re-added
            self.exclusion_list.discard(p["_id"])
            
        ids_np = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(embeddings.astype("float32"), ids_np)
        print(f"FAISS index added {len(products)} products. Total size: {self.index.ntotal}")

    def search(self, query: str, top_k=100) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
            
        embedding = self._embed_texts([query])
        
        # Adjust top_k if index has fewer elements
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
            
        scores, ids = self.index.search(embedding.astype("float32"), k)
        
        results = []
        for fid, score in zip(ids[0], scores[0]):
            fid_int = int(fid)
            if fid_int != -1 and fid_int in self.id_map:
                pid = self.id_map[fid_int]
                if pid in self.exclusion_list:
                    continue
                # Return score as standard float (dot product of normalized vectors = cosine)
                results.append((pid, float(score)))
                
        return results

    def save(self, index_path=None, map_path=None):
        if index_path is None or map_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_service_dir = os.path.dirname(current_dir)
            if index_path is None:
                index_path = os.path.join(search_service_dir, "models", "faiss.index")
            if map_path is None:
                map_path = os.path.join(search_service_dir, "models", "faiss_id_map.pkl")
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        with open(map_path, "wb") as f:
            pickle.dump((self.id_map, self.exclusion_list), f)
        print(f"FAISS index and ID map saved to {index_path}")

    def load(self, index_path=None, map_path=None):
        if index_path is None or map_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_service_dir = os.path.dirname(current_dir)
            if index_path is None:
                index_path = os.path.join(search_service_dir, "models", "faiss.index")
            if map_path is None:
                map_path = os.path.join(search_service_dir, "models", "faiss_id_map.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(map_path):
            print(f"FAISS index or ID map files not found at {index_path} or {map_path}")
            print(f"Current working directory: {os.getcwd()}")
            return False
            
        self.index = faiss.read_index(index_path)
        with open(map_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                self.id_map, self.exclusion_list = data
            else:
                self.id_map = data
                self.exclusion_list = set()
        print(f"FAISS index loaded from {index_path}. Total size: {self.index.ntotal}")
        return True
