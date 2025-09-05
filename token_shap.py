"""
Enhanced TokenSHAP implementation
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from collections import defaultdict
from functools import lru_cache
from scipy.special import comb
from transformers import AutoTokenizer, PreTrainedModel
from config import TokenSHAPConfig
from utils import TokenProcessor, ThreadSafeCache
from value_functions import ValueFunction, SimilarityValueFunction


class EnhancedTokenSHAP:
    """
    Enhanced TokenSHAP with improved sampling and convergence
    """
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: AutoTokenizer,
                 config: TokenSHAPConfig = None,
                 value_function: Optional[ValueFunction] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TokenSHAPConfig()
        self.processor = TokenProcessor(tokenizer)
        self.cache = ThreadSafeCache(self.config.cache_size) if self.config.cache_responses else None
        self.value_function = value_function or SimilarityValueFunction()
        
        # Move model to GPU if available
        if self.config.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
    
    @lru_cache(maxsize=1000)
    def _generate_response(self, prompt: str) -> str:
        """Generate response from model with caching"""
        if self.cache:
            cached = self.cache.get(prompt)
            if cached is not None:
                return cached
        
        # Encode and generate
        inputs = self.processor.encode(prompt, self.config.max_input_length)
        
        if self.config.use_gpu and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_output_length,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95
            )
        
        response = self.processor.decode(outputs)
        
        if self.cache:
            self.cache.set(prompt, response)
        
        return response
    
    def _stratify_tokens(self, tokens: List[str]) -> Dict[str, List[int]]:
        """Improved token stratification"""
        strata = defaultdict(list)
        
        for idx, token in enumerate(tokens):
            # Linguistic stratification
            if token.startswith('##'):  # Subword token
                strata['subword'].append(idx)
            elif token in ['.', ',', '!', '?', ';', ':']:
                strata['punctuation'].append(idx)
            elif token.isalpha():
                if len(token) > 6:
                    strata['long_words'].append(idx)
                elif len(token) <= 2:
                    strata['short_words'].append(idx)
                else:
                    strata['medium_words'].append(idx)
            elif token.isdigit():
                strata['numbers'].append(idx)
            else:
                strata['special'].append(idx)
        
        # Position-based stratification
        n = len(tokens)
        if n > 0:
            strata['position_start'] = list(range(min(3, n)))
            strata['position_end'] = list(range(max(0, n-3), n))
            
        return dict(strata)
    
    def _sample_subset_stratified(self, 
                                 tokens: List[str], 
                                 strata: Dict[str, List[int]], 
                                 subset_size: int) -> List[int]:
        """Fixed stratified sampling"""
        if subset_size >= len(tokens):
            return list(range(len(tokens)))
        
        if subset_size <= 0:
            return []
        
        # Calculate samples per stratum
        sampled_indices = set()
        non_empty_strata = [s for s in strata.values() if s]
        
        if not non_empty_strata:
            # Fallback to random sampling
            return list(np.random.choice(range(len(tokens)), subset_size, replace=False))
        
        tokens_per_stratum = max(1, subset_size // len(non_empty_strata))
        
        # Sample from each stratum
        for stratum_indices in non_empty_strata:
            n_sample = min(tokens_per_stratum, len(stratum_indices))
            sampled = np.random.choice(stratum_indices, n_sample, replace=False)
            sampled_indices.update(sampled)
        
        # Fill remaining slots
        sampled_list = list(sampled_indices)
        remaining = set(range(len(tokens))) - sampled_indices
        
        while len(sampled_list) < subset_size and remaining:
            additional = min(subset_size - len(sampled_list), len(remaining))
            sampled_list.extend(
                np.random.choice(list(remaining), additional, replace=False)
            )
            remaining = set(range(len(tokens))) - set(sampled_list)
        
        return sampled_list[:subset_size]
    
    def _compute_marginal_contribution(self,
                                      tokens: List[str],
                                      subset_indices: List[int],
                                      token_idx: int,
                                      full_response: str) -> float:
        """Compute marginal contribution of a token"""
        # Create prompts with and without the token
        with_token = [tokens[i] for i in subset_indices]
        without_token = [tokens[i] for i in subset_indices if i != token_idx]
        
        prompt_with = self.processor.reconstruct_from_tokens(with_token)
        prompt_without = self.processor.reconstruct_from_tokens(without_token)
        
        # Generate responses
        response_with = self._generate_response(prompt_with)
        response_without = self._generate_response(prompt_without)
        
        # Compute marginal contribution
        value_with = self.value_function.compute(full_response, response_with)
        value_without = self.value_function.compute(full_response, response_without)
        
        return value_with - value_without
    
    def compute_shapley_values(self, 
                              prompt: str, 
                              return_details: bool = False) -> Dict[str, float]:
        """
        Compute Shapley values with improved convergence
        """
        # Tokenize properly
        tokens = self.processor.tokenize(prompt)
        n_tokens = len(tokens)
        
        if n_tokens == 0:
            return {}
        
        # Initialize tracking
        shapley_values = np.zeros(n_tokens)
        sample_counts = np.zeros(n_tokens)
        
        # Generate full response
        full_response = self._generate_response(prompt)
        
        # Stage 1: First-order contributions
        if self.config.ensure_first_order:
            for i in range(n_tokens):
                subset_indices = [j for j in range(n_tokens) if j != i]
                contrib = self._compute_marginal_contribution(
                    tokens, subset_indices, i, full_response
                )
                shapley_values[i] += contrib
                sample_counts[i] += 1
        
        # Stratify tokens
        strata = self._stratify_tokens(tokens) if self.config.use_stratification else {}
        
        # Stage 2: Monte Carlo sampling with adaptive convergence
        converged = False
        iteration = 0
        convergence_history = []
        prev_values = shapley_values.copy()
        
        while iteration < self.config.max_samples and not converged:
            batch_updates = []
            
            # Process batch
            for _ in range(min(self.config.batch_size, self.config.max_samples - iteration)):
                # Sample subset size (weighted towards middle sizes)
                subset_size = min(n_tokens, max(1, int(np.random.beta(2, 2) * n_tokens)))
                
                # Sample subset
                if strata:
                    subset_indices = self._sample_subset_stratified(tokens, strata, subset_size)
                else:
                    subset_indices = list(np.random.choice(range(n_tokens), subset_size, replace=False))
                
                # Compute contributions for all tokens
                for token_idx in range(n_tokens):
                    if token_idx in subset_indices:
                        contrib = self._compute_marginal_contribution(
                            tokens, subset_indices, token_idx, full_response
                        )
                        # Weight by coalition size (Shapley kernel)
                        weight = self._shapley_kernel_weight(n_tokens, len(subset_indices))
                        batch_updates.append((token_idx, contrib * weight))
            
            # Update values
            for idx, weighted_contrib in batch_updates:
                shapley_values[idx] += weighted_contrib
                sample_counts[idx] += 1
            
            iteration += len(batch_updates) // n_tokens
            
            # Check convergence
            if self.config.adaptive_convergence and iteration >= self.config.min_samples:
                current_values = shapley_values / np.maximum(sample_counts, 1)
                max_change = np.max(np.abs(current_values - prev_values))
                convergence_history.append(max_change)
                
                if max_change < self.config.convergence_threshold:
                    if len(convergence_history) >= self.config.convergence_checks:
                        if all(c < self.config.convergence_threshold 
                              for c in convergence_history[-self.config.convergence_checks:]):
                            converged = True
                
                prev_values = current_values.copy()
        
        # Normalize
        final_values = shapley_values / np.maximum(sample_counts, 1)
        
        # Ensure efficiency axiom (values sum to v(N) - v(∅))
        v_full = self.value_function.compute(full_response, full_response)
        v_empty = 0.0  # Empty coalition value
        total_target = v_full - v_empty
        
        if np.sum(np.abs(final_values)) > 0:
            final_values = final_values * (total_target / np.sum(final_values))
        
        # Create result
        result = {token: float(final_values[i]) for i, token in enumerate(tokens)}
        
        if return_details:
            return {
                'shapley_values': result,
                'sample_counts': {tokens[i]: int(sample_counts[i]) for i in range(n_tokens)},
                'iterations': iteration,
                'converged': converged,
                'convergence_history': convergence_history,
                'cache_stats': self.cache.get_stats() if self.cache else None
            }
        
        return result
    
    def _shapley_kernel_weight(self, n: int, k: int) -> float:
        """Compute Shapley kernel weight for coalition of size k"""
        if k == 0 or k == n:
            return 1000.0  # Large weight for edge cases
        return (n - 1) / (comb(n, k) * k * (n - k))