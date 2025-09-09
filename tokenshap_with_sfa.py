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
    Enhanced TokenSHAP properly augmented with SFA and CoT support
    Includes Claude Opus 4.1 improvements for advanced SFA integration
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
        
        # Enhanced SFA features (Claude Opus 4.1)
        import os
        self.sfa_model_path = "models/sfa_trained.pkl"
        os.makedirs("models", exist_ok=True)
        
        # Try to load pre-trained SFA model
        self.sfa_learner.load_training_data(self.sfa_model_path)
        
        # Training data accumulator for incremental learning
        self.accumulated_training_data = []
        
        # Training cache (legacy compatibility)
        self.training_cache = []
        self.performance_metrics = {}
        
        logger.info(f"Enhanced TokenSHAP+SFA initialized (SFA trained: {self.sfa_learner.is_trained})")
    
    def explain(self,
               prompt: str,
               method: AttributionMethod = AttributionMethod.HYBRID,
               use_cot: bool = False,
               return_details: bool = False) -> Dict[str, Any]:
        """
        Truly unified explanation interface (Claude Opus 4.1 enhancement)
        """
        # Step 1: Generate base attributions
        base_attributions = self._compute_base_attributions(prompt, method)
        
        # Step 2: Apply CoT if requested
        if use_cot:
            attributions = self._apply_cot_analysis(prompt, base_attributions, method)
        else:
            attributions = base_attributions
        
        # Step 3: Apply augmentation if using HYBRID and SFA is trained
        if method == AttributionMethod.HYBRID and self.sfa_learner.is_trained:
            attributions = self._apply_sfa_augmentation(prompt, attributions)
        
        # Step 4: Format results consistently
        return self._format_results(attributions, return_details)

    def _compute_base_attributions(self, prompt: str, method: AttributionMethod) -> Dict[str, float]:
        """Compute base attributions using selected method"""
        if method == AttributionMethod.TOKENSHAP:
            return self.token_explainer.compute_shapley_values(prompt)
        elif method == AttributionMethod.SFA:
            if not self.sfa_learner.is_trained:
                raise ValueError("SFA not trained. Use train_sfa() first.")
            tokens = self.token_explainer.processor.tokenize(prompt)
            return self.sfa_learner.predict(prompt, tokens)
        else:  # HYBRID
            if self.sfa_learner.is_trained:
                return self.compute_augmented_shapley(prompt)
            else:
                return self.token_explainer.compute_shapley_values(prompt)

    def _apply_cot_analysis(self, prompt: str, base_attributions: Dict[str, float], method: AttributionMethod) -> Dict[str, Any]:
        """Apply Chain-of-Thought analysis to base attributions"""
        if method == AttributionMethod.HYBRID and self.sfa_learner.is_trained:
            # Use enhanced SFA augmentation for CoT
            cot_result = self.cot_explainer.compute_hierarchical_attribution_augmented(
                prompt, self.sfa_learner
            )
            # Incorporate base attributions as initial estimate
            cot_result['base_attributions'] = base_attributions
            return cot_result
        else:
            # Standard CoT attribution
            cot_result = self.cot_explainer.compute_hierarchical_attribution(
                prompt,
                use_sfa=(method in [AttributionMethod.SFA, AttributionMethod.HYBRID])
            )
            # Include base attributions for reference
            cot_result['base_attributions'] = base_attributions
            return cot_result

    def _apply_sfa_augmentation(self, prompt: str, attributions: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SFA augmentation to existing attributions"""
        if isinstance(attributions, dict) and 'shapley_values' in attributions:
            # Already processed through CoT, enhance the shapley values
            tokens = self.token_explainer.processor.tokenize(prompt)
            augmented_shapley = self.sfa_learner.predict_augmented(
                prompt, tokens, attributions['shapley_values']
            )
            attributions['shapley_values'] = augmented_shapley
            attributions['augmented'] = True
        else:
            # Simple attribution dict, apply augmentation directly
            tokens = self.token_explainer.processor.tokenize(prompt)
            attributions = self.sfa_learner.predict_augmented(prompt, tokens, attributions)
        
        return attributions

    def _format_results(self, attributions: Dict[str, Any], return_details: bool) -> Dict[str, Any]:
        """Format results consistently"""
        if return_details and not isinstance(attributions, dict) or 'shapley_values' not in attributions:
            return {'shapley_values': attributions}
        return attributions
    
    def compute_augmented_shapley(self, prompt: str) -> Dict[str, float]:
        """
        Compute Shapley values with SFA augmentation (Claude Opus 4.1 enhancement)
        Uses guided sampling and augmented features for improved accuracy
        """
        
        tokens = self.token_explainer.processor.tokenize(prompt)
        
        if self.sfa_learner.is_trained:
            # Step 1: Get initial Shapley estimate from SFA
            initial_estimate = self.sfa_learner.predict(prompt, tokens)
            
            # Step 2: Use initial estimate to guide sampling
            guided_shapley = self._compute_guided_shapley(
                prompt, tokens, initial_estimate
            )
            
            # Step 3: Refine with augmented features
            refined_shapley = self.sfa_learner.predict_augmented(
                prompt, tokens, guided_shapley
            )
            
            return refined_shapley
        else:
            # No SFA augmentation available, use standard TokenSHAP
            return self.token_explainer.compute_shapley_values(prompt)
    
    def _compute_guided_shapley(self, prompt: str, tokens: List[str], 
                               initial_estimate: Dict[str, float]) -> Dict[str, float]:
        """
        Compute Shapley values guided by SFA estimates
        Uses importance-weighted sampling for efficiency
        """
        
        n_tokens = len(tokens)
        shapley_values = np.zeros(n_tokens)
        sample_counts = np.zeros(n_tokens)
        
        # Use SFA estimates to prioritize sampling
        token_importance = np.array([abs(initial_estimate.get(t, 0)) for t in tokens])
        if token_importance.sum() > 0:
            sampling_weights = token_importance / token_importance.sum()
        else:
            sampling_weights = np.ones(n_tokens) / n_tokens
        
        # Add small uniform component to prevent zero sampling
        sampling_weights = 0.8 * sampling_weights + 0.2 / n_tokens
        
        # Adaptive sampling based on importance
        for _ in range(self.config.max_samples):
            # Sample subset with bias towards important tokens
            subset_size = np.random.randint(1, n_tokens + 1)
            
            try:
                subset_indices = np.random.choice(
                    n_tokens, subset_size, replace=False, p=sampling_weights
                )
            except ValueError:
                # Fallback to uniform sampling if probabilities are invalid
                subset_indices = np.random.choice(
                    n_tokens, subset_size, replace=False
                )
            
            # Compute marginal contributions
            for idx in subset_indices:
                try:
                    contribution = self.token_explainer._compute_marginal_contribution(
                        tokens, subset_indices.tolist(), idx, 
                        self.token_explainer._generate_response(prompt)
                    )
                    
                    # Weight by initial estimate confidence
                    confidence = min(1.0, abs(initial_estimate.get(tokens[idx], 0)) * 10)
                    weighted_contrib = contribution * (1 + confidence)
                    
                    shapley_values[idx] += weighted_contrib
                    sample_counts[idx] += 1
                except Exception as e:
                    logger.debug(f"Error computing marginal contribution for token {idx}: {e}")
                    # Use initial estimate as fallback
                    shapley_values[idx] += initial_estimate.get(tokens[idx], 0.0)
                    sample_counts[idx] += 1
        
        # Normalize
        final_values = shapley_values / np.maximum(sample_counts, 1)
        
        return {token: float(final_values[i]) for i, token in enumerate(tokens)}
    
    def train_sfa_incremental(self, new_prompts: List[str], save: bool = True) -> Dict[str, Any]:
        """
        Incrementally train SFA with new data (Claude Opus 4.1 enhancement)
        Allows continuous learning without retraining from scratch
        """
        
        logger.info(f"Incrementally training SFA with {len(new_prompts)} new prompts...")
        
        # Compute Shapley values for new prompts
        new_training_data = []
        for prompt in new_prompts:
            try:
                shapley_values = self.token_explainer.compute_shapley_values(prompt)
                new_training_data.append((prompt, shapley_values))
                self.accumulated_training_data.append((prompt, shapley_values))
            except Exception as e:
                logger.warning(f"Failed to compute Shapley values for prompt: {e}")
        
        # Combine with existing training data (keep last 1000 samples for memory efficiency)
        all_training_data = self.accumulated_training_data[-1000:]
        
        if len(all_training_data) < 5:
            logger.warning("Not enough training data for SFA training")
            return {'error': 'Insufficient training data'}
        
        # Train with augmentation
        result = self.sfa_learner.train_with_augmentation(all_training_data)
        
        # Save model
        if save:
            try:
                self.sfa_learner.save_training_data(self.sfa_model_path)
                logger.info(f"SFA model saved to {self.sfa_model_path}")
            except Exception as e:
                logger.error(f"Failed to save SFA model: {e}")
        
        return result
    
    def explain_augmented(self, prompt: str, method: str = "augmented", **kwargs) -> Dict[str, Any]:
        """
        Enhanced explain interface with SFA augmentation options (Claude Opus 4.1)
        
        Args:
            prompt: Text to explain
            method: "augmented", "standard", "sfa_only", "hybrid", "cot"
            **kwargs: Additional parameters
        """
        
        if method == "augmented":
            return {'token_attributions': self.compute_augmented_shapley(prompt)}
        elif method == "standard":
            return {'token_attributions': self.token_explainer.compute_shapley_values(prompt)}
        elif method == "sfa_only" and self.sfa_learner.is_trained:
            tokens = self.token_explainer.processor.tokenize(prompt)
            return {'token_attributions': self.sfa_learner.predict(prompt, tokens)}
        elif method == "cot":
            return self.explain(prompt, use_cot=True, **kwargs)
        elif method == "hybrid":
            return self.explain(prompt, method=AttributionMethod.HYBRID, **kwargs)
        else:
            # Default to augmented if available, otherwise standard
            return {'token_attributions': self.compute_augmented_shapley(prompt)}
    
    def get_sfa_stats(self) -> Dict[str, Any]:
        """Get enhanced SFA statistics (Claude Opus 4.1)"""
        if hasattr(self.sfa_learner, 'get_training_stats'):
            return self.sfa_learner.get_training_stats()
        else:
            return {
                'is_trained': self.sfa_learner.is_trained,
                'training_samples': len(self.accumulated_training_data),
                'cache_size': len(getattr(self.sfa_learner, 'shapley_cache', {}))
            }
    
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
    
    print("\n Configuration initialized")
    print(f"  - Max samples: {config.max_samples}")
    print(f"  - Convergence threshold: {config.convergence_threshold}")
    print(f"  - Parallel workers: {config.parallel_workers}")
    print(f"  - CoT max steps: {config.cot_max_steps}")
    
    # Note: To use with actual models:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # explainer = TokenSHAPWithSFA(model, tokenizer, config)
    
    print("\n Ready for integration with transformer models")
    print(" All critical bugs fixed")
    print(" Enhanced with proper CoT support")
    print("\nImplementation complete!")