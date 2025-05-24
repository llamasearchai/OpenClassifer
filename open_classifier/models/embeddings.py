"""Embedding model for semantic similarity and vector operations."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from pathlib import Path
import time

from ..core.config import settings
from ..core.logging import struct_logger
from ..core.exceptions import ModelLoadError

class EmbeddingModel:
    """Advanced embedding model with similarity search and clustering capabilities."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.logger = struct_logger.bind(component="embedding_model")
        
        # Create cache directory
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index for fast similarity search
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.text_store = []  # Store original texts
            
            # Cache for embeddings
            self._embedding_cache = {}
            self._cache_file = self.cache_dir / "embedding_cache.pkl"
            self._load_cache()
            
            self.logger.info("Embedding model initialized", 
                           model=self.model_name, 
                           dimension=self.embedding_dim)
            
        except Exception as e:
            self.logger.error("Failed to initialize embedding model", error=str(e))
            raise ModelLoadError(f"Failed to initialize embedding model: {str(e)}")
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                self.logger.info("Embedding cache loaded", cache_size=len(self._embedding_cache))
        except Exception as e:
            self.logger.warning("Failed to load embedding cache", error=str(e))
            self._embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self._cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
        except Exception as e:
            self.logger.warning("Failed to save embedding cache", error=str(e))
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    def encode(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Encode texts into embeddings with caching."""
        if not texts:
            return np.array([])
        
        embeddings = []
        texts_to_encode = []
        cached_indices = []
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if use_cache and text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
                cached_indices.append(i)
            else:
                texts_to_encode.append((i, text, text_hash))
        
        # Encode new texts
        if texts_to_encode:
            new_texts = [item[1] for item in texts_to_encode]
            new_embeddings = self.model.encode(new_texts, convert_to_numpy=True)
            
            # Cache new embeddings
            for (i, text, text_hash), embedding in zip(texts_to_encode, new_embeddings):
                if use_cache:
                    self._embedding_cache[text_hash] = embedding
                embeddings.insert(i - len([idx for idx in cached_indices if idx < i]), embedding)
        
        # Save cache periodically
        if len(texts_to_encode) > 0 and len(self._embedding_cache) % 100 == 0:
            self._save_cache()
        
        return np.array(embeddings)
    
    def find_similar(self, query_text: str, reference_texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar texts to the query."""
        if not reference_texts:
            return []
        
        try:
            # Encode query and reference texts
            query_embedding = self.encode([query_text])[0]
            reference_embeddings = self.encode(reference_texts)
            
            # Normalize for cosine similarity
            query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))[0]
            reference_embeddings = self._normalize_embeddings(reference_embeddings)
            
            # Calculate similarities
            similarities = np.dot(reference_embeddings, query_embedding)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "text": reference_texts[idx],
                    "similarity": float(similarities[idx]),
                    "index": int(idx)
                })
            
            return results
            
        except Exception as e:
            self.logger.error("Similarity search failed", error=str(e))
            return []
    
    def add_to_index(self, texts: List[str]):
        """Add texts to the FAISS index for fast similarity search."""
        if not texts:
            return
        
        try:
            # Encode texts
            embeddings = self.encode(texts)
            normalized_embeddings = self._normalize_embeddings(embeddings)
            
            # Add to FAISS index
            self.index.add(normalized_embeddings.astype('float32'))
            
            # Store original texts
            self.text_store.extend(texts)
            
            self.logger.info("Texts added to index", count=len(texts), total_in_index=len(self.text_store))
            
        except Exception as e:
            self.logger.error("Failed to add texts to index", error=str(e))
    
    def search_index(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index for similar texts."""
        if len(self.text_store) == 0:
            return []
        
        try:
            # Encode query
            query_embedding = self.encode([query_text])[0]
            normalized_query = self._normalize_embeddings(query_embedding.reshape(1, -1)).astype('float32')
            
            # Search index
            similarities, indices = self.index.search(normalized_query, min(top_k, len(self.text_store)))
            
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(self.text_store):
                    results.append({
                        "text": self.text_store[idx],
                        "similarity": float(sim),
                        "index": int(idx)
                    })
            
            return results
            
        except Exception as e:
            self.logger.error("Index search failed", error=str(e))
            return []
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster texts using K-means on embeddings."""
        if not texts or len(texts) < n_clusters:
            return {"error": "Not enough texts for clustering"}
        
        try:
            from sklearn.cluster import KMeans
            
            # Encode texts
            embeddings = self.encode(texts)
            normalized_embeddings = self._normalize_embeddings(embeddings)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_embeddings)
            
            # Organize results
            clusters = {}
            for i, (text, label) in enumerate(zip(texts, cluster_labels)):
                cluster_id = int(label)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                clusters[cluster_id].append({
                    "text": text,
                    "index": i
                })
            
            # Calculate cluster centers similarity to find representative texts
            for cluster_id, items in clusters.items():
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # Find most representative text (closest to centroid)
                best_similarity = -1
                representative_idx = 0
                
                for item in items:
                    similarity = np.dot(normalized_embeddings[item["index"]], cluster_center)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        representative_idx = item["index"]
                
                clusters[cluster_id] = {
                    "texts": items,
                    "representative_text": texts[representative_idx],
                    "representative_index": representative_idx,
                    "size": len(items)
                }
            
            return {
                "clusters": clusters,
                "n_clusters": n_clusters,
                "total_texts": len(texts)
            }
            
        except Exception as e:
            self.logger.error("Clustering failed", error=str(e))
            return {"error": f"Clustering failed: {str(e)}"}
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            embeddings = self.encode([text1, text2])
            normalized = self._normalize_embeddings(embeddings)
            
            similarity = np.dot(normalized[0], normalized[1])
            return float(similarity)
            
        except Exception as e:
            self.logger.error("Similarity calculation failed", error=str(e))
            return 0.0
    
    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text."""
        try:
            embedding = self.encode([text])[0]
            return embedding
        except Exception as e:
            self.logger.error("Failed to get text embedding", error=str(e))
            return None
    
    def batch_similarity(self, query_texts: List[str], reference_texts: List[str]) -> np.ndarray:
        """Calculate similarity matrix between query and reference texts."""
        try:
            query_embeddings = self.encode(query_texts)
            reference_embeddings = self.encode(reference_texts)
            
            query_normalized = self._normalize_embeddings(query_embeddings)
            reference_normalized = self._normalize_embeddings(reference_embeddings)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(query_normalized, reference_normalized.T)
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error("Batch similarity calculation failed", error=str(e))
            return np.array([])
    
    def save_index(self, filepath: str):
        """Save the FAISS index to disk."""
        try:
            faiss.write_index(self.index, filepath)
            
            # Save text store
            text_store_path = filepath.replace('.index', '_texts.pkl')
            with open(text_store_path, 'wb') as f:
                pickle.dump(self.text_store, f)
            
            self.logger.info("Index saved", filepath=filepath)
            
        except Exception as e:
            self.logger.error("Failed to save index", error=str(e))
    
    def load_index(self, filepath: str):
        """Load FAISS index from disk."""
        try:
            self.index = faiss.read_index(filepath)
            
            # Load text store
            text_store_path = filepath.replace('.index', '_texts.pkl')
            if os.path.exists(text_store_path):
                with open(text_store_path, 'rb') as f:
                    self.text_store = pickle.load(f)
            
            self.logger.info("Index loaded", filepath=filepath, texts_count=len(self.text_store))
            
        except Exception as e:
            self.logger.error("Failed to load index", error=str(e))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "cache_size": len(self._embedding_cache),
            "index_size": len(self.text_store),
            "cache_file": str(self._cache_file)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()
        self.logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Save cache when object is destroyed."""
        try:
            self._save_cache()
        except:
            pass 