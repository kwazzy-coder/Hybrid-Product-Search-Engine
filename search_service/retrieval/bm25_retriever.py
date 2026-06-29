import os
import pickle
import re
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.index = None          # BM25Okapi instance
        self.product_ids = []      # parallel list — index i → product_id

    def build(self, products: list[dict]):
        if not products:
            print("No products provided to build BM25 index.")
            return
        # tokenise embedding_text field for each product
        corpus = [self.tokenise(p.get("embedding_text", "")) for p in products]
        self.product_ids = [p["_id"] for p in products]
        self.index = BM25Okapi(corpus)
        print(f"BM25 index successfully built with {len(self.product_ids)} products.")

    def tokenise(self, text: str) -> list[str]:
        if not text:
            return []
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).split()

    def search(self, expanded_terms: list[str], top_k=100) -> list[tuple[str, float]]:
        if not self.index or not self.product_ids:
            return []
        # join expanded terms as one query
        query_tokens = self.tokenise(" ".join(expanded_terms))
        if not query_tokens:
            return []
            
        scores = self.index.get_scores(query_tokens)
        
        # Argsort scores to get indices of top scores in descending order
        # Using partition or argsort
        import numpy as np
        scores_arr = np.array(scores)
        
        # If we have fewer candidates than top_k, adjust top_k
        k = min(top_k, len(scores_arr))
        if k == 0:
            return []
            
        # Get top k indices
        top_indices = np.argsort(scores_arr)[::-1][:k]
        
        return [(self.product_ids[i], float(scores_arr[i])) for i in top_indices]

    def save(self, path=None):
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_service_dir = os.path.dirname(current_dir)
            path = os.path.join(search_service_dir, "models", "bm25.pkl")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self.index, self.product_ids), f)
        print(f"BM25 index saved to {path}.")

    def load(self, path=None):
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_service_dir = os.path.dirname(current_dir)
            path = os.path.join(search_service_dir, "models", "bm25.pkl")
        
        if not os.path.exists(path):
            print(f"BM25 index file not found at {path}.")
            print(f"Current working directory: {os.getcwd()}")
            return False
        with open(path, "rb") as f:
            self.index, self.product_ids = pickle.load(f)
        print(f"BM25 index loaded from {path} with {len(self.product_ids)} products.")
        return True
