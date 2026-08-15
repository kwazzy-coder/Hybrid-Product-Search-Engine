import os
import pickle
import faiss
import numpy as np

class CLIPRetriever:
    def __init__(self):
        self.embedding_dim = 512
        self.model = None
        self.processor = None
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        self.id_map = {}
        self.exclusion_list = set()
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            self.model.eval()
            print('CLIP model loaded for visual search.')
        except Exception as e:
            print(f'CLIP model unavailable: {e}. Visual search disabled.')

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.model is None or self.processor is None:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        import torch
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().astype(np.float32)

    def encode_images(self, images) -> np.ndarray:
        if self.model is None or self.processor is None:
            return np.zeros((len(images), self.embedding_dim), dtype=np.float32)
        
        import torch
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().astype(np.float32)

    def search(self, query: str, top_k=100) -> list[tuple[str, float]]:
        if self.model is None:
            return []
            
        if self.index.ntotal == 0:
            return []
            
        embedding = self.encode_texts([query])
        
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
            
        scores, ids = self.index.search(embedding, k)
        
        results = []
        for fid, score in zip(ids[0], scores[0]):
            fid_int = int(fid)
            if fid_int != -1 and fid_int in self.id_map:
                pid = self.id_map[fid_int]
                if pid in self.exclusion_list:
                    continue
                results.append((pid, float(score)))
                
        return results

    def add_image_embeddings(self, embeddings: np.ndarray, product_ids: list[str]):
        if len(product_ids) == 0:
            return
            
        ids = []
        for pid in product_ids:
            fid = int(hash(pid) & 0x7FFFFFFFFFFFFFFF)
            ids.append(fid)
            self.id_map[fid] = pid
            self.exclusion_list.discard(pid)
            
        ids_np = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(embeddings.astype("float32"), ids_np)
        print(f"CLIP index added {len(product_ids)} items. Total size: {self.index.ntotal}")

    def save(self, index_path=None, map_path=None):
        if index_path is None or map_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_service_dir = os.path.dirname(current_dir)
            if index_path is None:
                index_path = os.path.join(search_service_dir, "models", "clip.index")
            if map_path is None:
                map_path = os.path.join(search_service_dir, "models", "clip_id_map.pkl")
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        with open(map_path, "wb") as f:
            pickle.dump((self.id_map, self.exclusion_list), f)
        print(f"CLIP index and ID map saved to {index_path}")

    def load(self, index_path=None, map_path=None):
        if index_path is None or map_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_service_dir = os.path.dirname(current_dir)
            if index_path is None:
                index_path = os.path.join(search_service_dir, "models", "clip.index")
            if map_path is None:
                map_path = os.path.join(search_service_dir, "models", "clip_id_map.pkl")
        
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
