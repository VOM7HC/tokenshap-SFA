"""
Value function classes for TokenSHAP
"""

import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, PreTrainedModel


class ValueFunction(ABC):
    """Abstract base class for value functions"""
    
    @abstractmethod
    def compute(self, original: str, subset: str) -> float:
        """Compute value for subset relative to original"""
        pass


class SimilarityValueFunction(ValueFunction):
    """TF-IDF cosine similarity value function"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        
    def compute(self, original: str, subset: str) -> float:
        """Compute cosine similarity between original and subset responses"""
        if not original or not subset:
            return 0.0
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([original, subset])
            similarity = 1 - cosine(
                tfidf_matrix[0].toarray()[0], 
                tfidf_matrix[1].toarray()[0]
            )
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except:
            return 0.0


class ProbabilityValueFunction(ValueFunction):
    """Log probability based value function"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def compute(self, original: str, subset: str) -> float:
        """Compute value based on model's log probability"""
        if not subset:
            return 0.0
            
        inputs = self.tokenizer(subset, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Compute average log probability
            log_probs = F.log_softmax(logits, dim=-1)
            avg_log_prob = log_probs.mean().item()
            
            # Normalize to [0, 1]
            return 1.0 / (1.0 + np.exp(-avg_log_prob))