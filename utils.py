"""
Utility classes for TokenSHAP with SFA
"""

from collections import OrderedDict
from typing import List, Dict, Optional, Any
import hashlib
import threading
import torch
from transformers import AutoTokenizer


class ThreadSafeCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[Any]:
        """Retrieve cached response with thread safety"""
        key = self._get_key(prompt)
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, prompt: str, response: Any):
        """Store response in cache with proper LRU eviction"""
        key = self._get_key(prompt)
        with self.lock:
            if key in self.cache:
                # Update and move to end
                self.cache.move_to_end(key)
            else:
                # Add new entry
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
            self.cache[key] = response
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


class TokenProcessor:
    """Handles tokenization and token processing"""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        
    def tokenize(self, text: str) -> List[str]:
        """Properly tokenize text using model tokenizer"""
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to tensor"""
        return self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode tensor to text"""
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
    
    def reconstruct_from_tokens(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens"""
        # Handle subword tokens properly
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text.strip()