import os
import lightgbm as lgb
import numpy as np

class LTRModel:
    def __init__(self, model_path=None):
        if model_path is None:
            # Construct path relative to this file's location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_service_dir = os.path.dirname(current_dir)
            model_path = os.path.join(search_service_dir, "models", "ltr_model.lgb")
        self.model_path = model_path
        self.model = None
        self.load()

    def load(self):
        if os.path.exists(self.model_path):
            try:
                self.model = lgb.Booster(model_file=self.model_path)
                print(f"LightGBM LTR model successfully loaded from {self.model_path}.")
                return True
            except Exception as e:
                print(f"Error loading LTR model: {e}. Reranking will fallback to RRF sorting.")
        else:
            print(f"Warning: LTR model file not found at {self.model_path}. Reranking will fallback to RRF sorting.")
            print(f"Current working directory: {os.getcwd()}")
        return False

    def predict(self, feature_vectors: list[list[float]]) -> list[float]:
        if not feature_vectors:
            return []
            
        if self.model:
            try:
                # Predict using LightGBM model
                preds = self.model.predict(np.array(feature_vectors, dtype=np.float32))
                return [float(p) for p in preds]
            except Exception as e:
                print(f"Prediction error: {e}. Falling back to RRF sorting.")
                
        # Fallback: RRF score is the first feature (f1) in our feature vector
        return [float(fv[0]) for fv in feature_vectors]
