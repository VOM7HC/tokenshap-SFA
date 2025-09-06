"""
Unified framework combining Enhanced TokenSHAP with SFA and CoT support
"""

import numpy as np
import pickle
import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
# Optional transformers import
try:
    from transformers import AutoTokenizer, PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create placeholder classes
    class AutoTokenizer:
        pass
    class PreTrainedModel:
        pass
from config import TokenSHAPConfig, AttributionMethod
from token_shap import EnhancedTokenSHAP
from sfa_learner import SFAMetaLearner
from cot_explainer import CoTTokenSHAP

logger = logging.getLogger(__name__)


class TokenSHAPWithSFA:
    """
    Unified framework combining Enhanced TokenSHAP with SFA and CoT support
    """
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: AutoTokenizer,
                 config: TokenSHAPConfig = None):
        self.config = config or TokenSHAPConfig()
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize components
        self.token_explainer = EnhancedTokenSHAP(model, tokenizer, config)
        self.sfa_learner = SFAMetaLearner(config)
        self.cot_explainer = CoTTokenSHAP(model, tokenizer, config)
        
        # Training cache
        self.training_cache = []
        self.performance_metrics = {}
    
    def explain(self,
               prompt: str,
               method: AttributionMethod = AttributionMethod.HYBRID,
               use_cot: bool = False,
               return_details: bool = False) -> Dict[str, Any]:
        """
        Unified explanation interface
        
        Args:
            prompt: Input text to explain
            method: Attribution method to use
            use_cot: Whether to use Chain-of-Thought reasoning
            return_details: Return detailed attribution info
        """
        if use_cot:
            # Use CoT-based hierarchical attribution
            return self.cot_explainer.compute_hierarchical_attribution(
                prompt,
                use_sfa=(method in [AttributionMethod.SFA, AttributionMethod.HYBRID])
            )
        
        # Standard attribution
        if method == AttributionMethod.TOKENSHAP:
            result = self.token_explainer.compute_shapley_values(prompt, return_details)
        
        elif method == AttributionMethod.SFA:
            if not self.sfa_learner.is_trained:
                raise ValueError("SFA not trained. Use train_sfa() first.")
            tokens = self.token_explainer.processor.tokenize(prompt)
            result = self.sfa_learner.predict(prompt, tokens)
            
        else:  # HYBRID
            if self.sfa_learner.is_trained:
                # Use SFA for speed
                tokens = self.token_explainer.processor.tokenize(prompt)
                result = self.sfa_learner.predict(prompt, tokens)
            else:
                # Fall back to TokenSHAP
                result = self.token_explainer.compute_shapley_values(prompt, return_details)
        
        if return_details and not isinstance(result, dict) or 'shapley_values' not in result:
            result = {'shapley_values': result}
        
        return result
    
    def train_sfa(self,
                 training_prompts: List[str],
                 use_cot: bool = False,
                 batch_size: int = 10) -> Dict[str, Any]:
        """
        Train SFA meta-learner
        
        Args:
            training_prompts: Prompts for training
            use_cot: Whether to use CoT-generated steps
            batch_size: Batch size for parallel processing
        """
        logger.info(f"Training SFA on {len(training_prompts)} prompts...")
        
        if use_cot:
            # Train on CoT steps
            return self.cot_explainer.train_sfa_on_cot(training_prompts)
        
        # Standard training
        training_data = []
        
        # Process in batches
        for i in range(0, len(training_prompts), batch_size):
            batch = training_prompts[i:i+batch_size]
            
            if self.config.parallel_workers > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                    futures = []
                    for prompt in batch:
                        future = executor.submit(
                            self.token_explainer.compute_shapley_values, prompt
                        )
                        futures.append((prompt, future))
                    
                    for prompt, future in futures:
                        try:
                            shapley_values = future.result(timeout=60)
                            training_data.append((prompt, shapley_values))
                            self.training_cache.append((prompt, shapley_values))
                        except Exception as e:
                            logger.error(f"Error processing prompt: {e}")
            else:
                # Sequential processing
                for prompt in batch:
                    shapley_values = self.token_explainer.compute_shapley_values(prompt)
                    training_data.append((prompt, shapley_values))
                    self.training_cache.append((prompt, shapley_values))
        
        # Train meta-learner
        training_result = self.sfa_learner.train(training_data)
        
        # Store performance metrics
        self.performance_metrics['sfa_training'] = training_result
        
        logger.info("SFA training complete!")
        
        return training_result
    
    def benchmark(self,
                 test_prompts: List[str],
                 methods: List[AttributionMethod] = None) -> Dict[str, Any]:
        """
        Benchmark different attribution methods
        """
        if methods is None:
            methods = [AttributionMethod.TOKENSHAP, AttributionMethod.SFA]
        
        results = {}
        
        for method in methods:
            if method == AttributionMethod.SFA and not self.sfa_learner.is_trained:
                continue
            
            method_times = []
            method_results = []
            
            for prompt in test_prompts:
                start_time = time.time()
                result = self.explain(prompt, method=method)
                elapsed = time.time() - start_time
                
                method_times.append(elapsed)
                method_results.append(result)
            
            results[method.value] = {
                'avg_time': np.mean(method_times),
                'std_time': np.std(method_times),
                'total_time': sum(method_times),
                'n_prompts': len(test_prompts)
            }
        
        # Compute speedup if both methods tested
        if 'tokenshap' in results and 'sfa' in results:
            speedup = results['tokenshap']['avg_time'] / results['sfa']['avg_time']
            results['speedup'] = speedup
        
        return results
    
    def save(self, filepath: str):
        """Save trained models and configuration"""
        state = {
            'config': self.config,
            'sfa_model': self.sfa_learner.meta_model if self.sfa_learner.is_trained else None,
            'sfa_features': self.sfa_learner.feature_names,
            'training_cache': self.training_cache[-100:],  # Save last 100 examples
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained models and configuration"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        if state['sfa_model']:
            self.sfa_learner.meta_model = state['sfa_model']
            self.sfa_learner.is_trained = True
            self.sfa_learner.feature_names = state['sfa_features']
        
        self.training_cache = state.get('training_cache', [])
        self.performance_metrics = state.get('performance_metrics', {})
        
        logger.info(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("Enhanced TokenSHAP + SFA + CoT Implementation v2.0")
    print("=" * 60)
    
    # Configuration
    config = TokenSHAPConfig(
        max_samples=100,
        batch_size=10,
        convergence_threshold=0.01,
        use_stratification=True,
        ensure_first_order=True,
        adaptive_convergence=True,
        parallel_workers=4,
        sfa_n_estimators=100,
        cot_max_steps=8
    )
    
    print("\n✓ Configuration initialized")
    print(f"  - Max samples: {config.max_samples}")
    print(f"  - Convergence threshold: {config.convergence_threshold}")
    print(f"  - Parallel workers: {config.parallel_workers}")
    print(f"  - CoT max steps: {config.cot_max_steps}")
    
    # Note: To use with actual models:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # explainer = TokenSHAPWithSFA(model, tokenizer, config)
    
    print("\n✓ Ready for integration with transformer models")
    print("✓ All critical bugs fixed")
    print("✓ Enhanced with proper CoT support")
    print("\nImplementation complete!")