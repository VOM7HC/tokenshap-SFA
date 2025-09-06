"""
TokenSHAP with Ollama model support
"""

import numpy as np
import pickle
import logging
import time
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from config import TokenSHAPConfig, AttributionMethod
from ollama_integration import OllamaModel, SimpleOllamaModel, OllamaModelAdapter
from sfa_learner import SFAMetaLearner

# Try to import transformers, but make it optional
try:
    from transformers import AutoTokenizer, PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create mock classes for type hints
    class AutoTokenizer: pass
    class PreTrainedModel: pass

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Simple tokenizer for Ollama models when transformers is not available"""
    
    def __init__(self, model_name: str = "simple"):
        self.model_name = model_name
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word-based tokenization"""
        # Basic tokenization - split on spaces and punctuation
        import re
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return tokens
    
    def encode(self, text: str, return_tensors: str = "pt", 
               truncation: bool = True, max_length: int = 512, 
               padding: bool = True) -> Dict[str, Any]:
        """Simple encoding that returns text-based representation"""
        tokens = self.tokenize(text)
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return {
            'input_ids': tokens,  # Keep as tokens for Ollama
            'attention_mask': [1] * len(tokens)
        }
    
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Simple decoding"""
        if isinstance(token_ids, list):
            return ' '.join(str(t) for t in token_ids if str(t) not in ['[PAD]', '[EOS]'] or not skip_special_tokens)
        return str(token_ids)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string"""
        return ' '.join(tokens)


class OllamaTokenProcessor:
    """Token processor for Ollama models"""
    
    def __init__(self, tokenizer: Union[AutoTokenizer, SimpleTokenizer]):
        self.tokenizer = tokenizer
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return self.tokenizer.tokenize(text)
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """Encode text"""
        return self.tokenizer.encode(text, max_length=max_length, truncation=True, padding=True)
    
    def decode(self, token_ids) -> str:
        """Decode tokens"""
        if hasattr(token_ids, '__getitem__') and len(token_ids) > 0:
            # Handle list or tensor-like objects
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return str(token_ids)
    
    def reconstruct_from_tokens(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens"""
        return self.tokenizer.convert_tokens_to_string(tokens)


class OllamaTokenSHAP:
    """TokenSHAP implementation optimized for Ollama models"""
    
    def __init__(self, 
                 model: Union[OllamaModel, SimpleOllamaModel],
                 tokenizer: Union[AutoTokenizer, SimpleTokenizer] = None,
                 config: TokenSHAPConfig = None):
        
        self.model = model
        self.config = config or TokenSHAPConfig()
        
        # Use simple tokenizer if none provided
        if tokenizer is None:
            tokenizer = SimpleTokenizer()
        
        self.tokenizer = tokenizer
        self.processor = OllamaTokenProcessor(tokenizer)
        
        # Simple cache for responses
        self.response_cache = {}
        
    def _generate_response(self, prompt: str) -> str:
        """Generate response using Ollama model"""
        # Check cache first
        if prompt in self.response_cache:
            return self.response_cache[prompt]
        
        try:
            response = self.model.generate(
                prompt=prompt,
                max_length=self.config.max_output_length,
                temperature=self.config.temperature
            )
            
            # Cache the response
            if len(self.response_cache) < self.config.cache_size:
                self.response_cache[prompt] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_shapley_values(self, prompt: str, max_samples: int = None) -> Dict[str, float]:
        """Compute Shapley values for tokens in the prompt"""
        if max_samples is None:
            max_samples = self.config.max_samples
        
        # Tokenize the prompt
        tokens = self.processor.tokenize(prompt)
        n_tokens = len(tokens)
        
        if n_tokens == 0:
            return {}
        
        # Generate full response for reference
        full_response = self._generate_response(prompt)
        
        # Initialize Shapley values
        shapley_values = np.zeros(n_tokens)
        sample_counts = np.zeros(n_tokens)
        
        logger.info(f"Computing Shapley values for {n_tokens} tokens with {max_samples} samples")
        
        # Monte Carlo sampling
        for iteration in range(max_samples):
            # Sample subset size
            subset_size = np.random.randint(1, n_tokens + 1)
            
            # Sample subset of tokens
            subset_indices = np.random.choice(n_tokens, subset_size, replace=False)
            
            # For each token in the subset, compute marginal contribution
            for token_idx in subset_indices:
                # Create subset without this token
                other_indices = [idx for idx in subset_indices if idx != token_idx]
                
                # Generate responses
                with_token = [tokens[i] for i in subset_indices]
                without_token = [tokens[i] for i in other_indices]
                
                prompt_with = self.processor.reconstruct_from_tokens(with_token)
                prompt_without = self.processor.reconstruct_from_tokens(without_token)
                
                response_with = self._generate_response(prompt_with)
                response_without = self._generate_response(prompt_without)
                
                # Compute marginal contribution
                similarity_with = self._compute_similarity(full_response, response_with)
                similarity_without = self._compute_similarity(full_response, response_without)
                
                contribution = similarity_with - similarity_without
                
                # Update Shapley value
                shapley_values[token_idx] += contribution
                sample_counts[token_idx] += 1
            
            # Progress logging
            if (iteration + 1) % 10 == 0:
                logger.info(f"Completed {iteration + 1}/{max_samples} samples")
        
        # Average the values
        final_values = np.divide(shapley_values, sample_counts, 
                               out=np.zeros_like(shapley_values), 
                               where=sample_counts!=0)
        
        # Create result dictionary
        result = {token: float(final_values[i]) for i, token in enumerate(tokens)}
        
        return result


class TokenSHAPWithOllama:
    """Main interface for TokenSHAP with Ollama support"""
    
    def __init__(self,
                 model_name: str,
                 api_url: str = "http://127.0.0.1:11434",
                 tokenizer: Optional[Union[AutoTokenizer, SimpleTokenizer]] = None,
                 config: TokenSHAPConfig = None,
                 use_simple_model: bool = False):
        
        self.config = config or TokenSHAPConfig()
        
        # Create Ollama model
        if use_simple_model:
            self.model = SimpleOllamaModel(model_name, api_url)
        else:
            self.model = OllamaModel(model_name, api_url)
        
        # Setup tokenizer
        if tokenizer is None:
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Try to use a compatible tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except:
                    self.tokenizer = SimpleTokenizer()
            else:
                self.tokenizer = SimpleTokenizer()
        else:
            self.tokenizer = tokenizer
        
        # Initialize TokenSHAP
        self.token_explainer = OllamaTokenSHAP(self.model, self.tokenizer, self.config)
        
        # Initialize SFA learner
        self.sfa_learner = SFAMetaLearner(self.config)
        
        # Cache for training data
        self.training_cache = []
        
        logger.info(f"Initialized TokenSHAP with Ollama model: {model_name}")
        
    def explain(self, prompt: str, method: str = "tokenshap", **kwargs) -> Dict[str, Any]:
        """Explain a prompt using TokenSHAP"""
        
        if method.lower() == "sfa" and self.sfa_learner.is_trained:
            # Use SFA for fast prediction
            tokens = self.token_explainer.processor.tokenize(prompt)
            return self.sfa_learner.predict(prompt, tokens)
        
        elif method.lower() in ["tokenshap", "shapley"]:
            # Use full TokenSHAP computation
            max_samples = kwargs.get('max_samples', self.config.max_samples)
            return self.token_explainer.compute_shapley_values(prompt, max_samples)
        
        else:
            # Default to TokenSHAP
            return self.token_explainer.compute_shapley_values(prompt)
    
    def train_sfa(self, training_prompts: List[str], batch_size: int = 5) -> Dict[str, Any]:
        """Train SFA meta-learner on the given prompts"""
        logger.info(f"Training SFA on {len(training_prompts)} prompts...")
        
        training_data = []
        
        for i, prompt in enumerate(training_prompts):
            logger.info(f"Processing training prompt {i+1}/{len(training_prompts)}")
            
            # Compute Shapley values
            shapley_values = self.token_explainer.compute_shapley_values(
                prompt, max_samples=min(20, self.config.max_samples)  # Fewer samples for training
            )
            
            training_data.append((prompt, shapley_values))
            self.training_cache.append((prompt, shapley_values))
        
        # Train the SFA learner
        training_result = self.sfa_learner.train(training_data)
        logger.info("SFA training completed!")
        
        return training_result
    
    def benchmark(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark TokenSHAP vs SFA performance"""
        results = {}
        
        # Test TokenSHAP
        tokenshap_times = []
        for prompt in test_prompts:
            start_time = time.time()
            self.explain(prompt, method="tokenshap")
            elapsed = time.time() - start_time
            tokenshap_times.append(elapsed)
        
        results['tokenshap'] = {
            'avg_time': np.mean(tokenshap_times),
            'total_time': sum(tokenshap_times)
        }
        
        # Test SFA if trained
        if self.sfa_learner.is_trained:
            sfa_times = []
            for prompt in test_prompts:
                start_time = time.time()
                self.explain(prompt, method="sfa")
                elapsed = time.time() - start_time
                sfa_times.append(elapsed)
            
            results['sfa'] = {
                'avg_time': np.mean(sfa_times),
                'total_time': sum(sfa_times)
            }
            
            # Calculate speedup
            results['speedup'] = results['tokenshap']['avg_time'] / results['sfa']['avg_time']
        
        return results
    
    def save(self, filepath: str):
        """Save the trained model"""
        state = {
            'config': self.config,
            'model_name': self.model.model_name,
            'api_url': self.model.api_url,
            'sfa_model': self.sfa_learner.meta_model if self.sfa_learner.is_trained else None,
            'sfa_features': self.sfa_learner.feature_names,
            'training_cache': self.training_cache[-100:]  # Save last 100
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        if state['sfa_model']:
            self.sfa_learner.meta_model = state['sfa_model']
            self.sfa_learner.is_trained = True
            self.sfa_learner.feature_names = state['sfa_features']
        
        self.training_cache = state.get('training_cache', [])
        logger.info(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("TokenSHAP with Ollama Integration")
    print("=" * 40)
    
    # Using phi4-reasoning model (GPU optimized)
    models_to_test = [
        {"name": "phi4-reasoning:latest", "url": "http://127.0.0.1:11434"},
    ]
    
    for model_config in models_to_test:
        print(f"\nTesting model: {model_config['name']}")
        print("💡 phi4-reasoning (14.7B parameters) - GPU accelerated")
        print("⏰ Expected time: 1-2 minutes with warmup")
        
        try:
            # Initialize explainer with phi4-reasoning optimizations
            print("🔄 Initializing TokenSHAP with phi4-reasoning...")
            explainer = TokenSHAPWithOllama(
                model_name=model_config["name"],
                api_url=model_config["url"],
                config=TokenSHAPConfig(
                    max_samples=3,  # Reduced for phi4-reasoning performance
                    parallel_workers=1,  # Single worker for stability
                    convergence_threshold=0.1  # Less strict for faster results
                )
            )
            
            # Test explanation
            test_prompt = "The quick brown fox jumps over the lazy dog."
            print(f"⚡ Analyzing: '{test_prompt}'")
            print("🔄 Processing with phi4-reasoning (please be patient)...")
            
            result = explainer.explain(test_prompt)
            
            print("✅ Analysis completed!")
            print("📊 Top tokens by importance:")
            sorted_tokens = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
            for token, importance in sorted_tokens[:5]:
                print(f"  {token}: {importance:.4f}")
                
        except Exception as e:
            print(f"❌ Error with {model_config['name']}: {str(e)}")
            if "timeout" in str(e).lower():
                print("💡 phi4-reasoning timed out - this is normal for large models")
            else:
                print("💡 Make sure Ollama is running and model is available")
    
    print("\n✓ Example completed!")