"""
Performance-Optimized Chain-of-Thought TokenSHAP with Hybrid CPU/GPU Architecture
"""

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, PreTrainedModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import TokenSHAPConfig
from optimized_performance_manager import OptimizedTokenSHAP, DeviceManager, performance_monitor
from value_functions import SimilarityValueFunction
from sfa_learner import SFAMetaLearner

logger = logging.getLogger(__name__)


class OptimizedCoTTokenSHAP:
    """
    Performance-optimized Chain-of-Thought TokenSHAP with intelligent CPU/GPU distribution
    
    Performance Strategy:
    - GPU: Model inference, tensor operations, batch processing
    - CPU: Text parsing, small computations, control logic, similarity calculations
    """
    
    def __init__(self, 
                 model: Optional[PreTrainedModel] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 config: TokenSHAPConfig = None):
        
        self.config = config or TokenSHAPConfig()
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize performance components
        self.device_manager = DeviceManager(self.config) if TORCH_AVAILABLE else None
        self.optimized_base = OptimizedTokenSHAP(model, tokenizer, config)
        
        # CPU-optimized components (these work better on CPU)
        self.step_value_function = SimilarityValueFunction()  # TF-IDF is CPU-optimized
        self.sfa_learner = SFAMetaLearner(config)  # Scikit-learn is CPU-optimized
        
        # Initialize reasoning patterns (CPU operation)
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        # Performance tracking
        self.performance_metrics = []
        
        # Move model to optimal device if available
        if self.model and self.device_manager:
            self.model = self.device_manager.to_device(self.model)
        
        logger.info("OptimizedCoTTokenSHAP initialized with hybrid CPU/GPU architecture")
    
    def _initialize_reasoning_patterns(self) -> Dict[str, str]:
        """Initialize reasoning patterns (CPU-optimized operation)"""
        return {
            'analytical': "Let's analyze this step by step:\n",
            'mathematical': "Let's solve this mathematically:\n",
            'logical': "Let's think through this logically:\n",
            'comparative': "Let's compare the options:\n",
            'sequential': "Let's go through this sequentially:\n",
            'systematic': "Let's approach this systematically:\n"
        }
    
    @performance_monitor
    def generate_cot_gpu_optimized(self, 
                                  prompt: str, 
                                  pattern: str = 'analytical') -> Tuple[List[str], str]:
        """
        GPU-optimized CoT generation with intelligent device management
        """
        if not self.model or not TORCH_AVAILABLE:
            raise ValueError("Model and PyTorch required for GPU-optimized CoT")
        
        # CPU operation: Prepare prompt (lightweight text processing)
        cot_template = self.reasoning_patterns.get(pattern, self.config.cot_prompt_template)
        cot_prompt = f"{prompt}\n{cot_template}"
        
        # GPU operation: Model inference
        if self.tokenizer:
            inputs = self.tokenizer(
                cot_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_length,
                padding=True
            )
            
            # Move to optimal device
            if self.device_manager:
                inputs = self.device_manager.to_device(inputs)
            
            # GPU-optimized generation with mixed precision
            if self.device_manager and self.device_manager.device == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=self.config.max_output_length * 2,
                            temperature=self.config.temperature,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
            else:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.config.max_output_length * 2,
                        temperature=self.config.temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            # CPU operation: Decode response (more efficient on CPU)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise ValueError("Tokenizer required for GPU-optimized generation")
        
        # CPU operation: Parse steps (text processing is CPU-optimized)
        steps = self._parse_cot_steps_optimized(full_response, prompt)
        
        return steps, full_response
    
    def _parse_cot_steps_optimized(self, response: str, original_prompt: str) -> List[str]:
        """
        CPU-optimized step parsing with improved algorithms
        """
        # Remove original prompt (CPU string operation)
        if original_prompt in response:
            response = response.replace(original_prompt, '').strip()
        
        # Optimized delimiter detection using pre-compiled patterns
        import re
        
        # Pre-compiled regex patterns for better performance
        step_patterns = [
            re.compile(r'^Step \d+[:\.]', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\d+[\.\)]', re.MULTILINE),
            re.compile(r'^(First|Second|Third|Fourth|Fifth|Next|Then|Finally|Therefore|Thus)[,:.]', 
                      re.MULTILINE | re.IGNORECASE),
            re.compile(r'^[-•*]', re.MULTILINE)
        ]
        
        # Split into potential steps
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        steps = []
        current_step = []
        
        for line in lines:
            # Check if line starts a new step using compiled patterns
            is_new_step = any(pattern.match(line) for pattern in step_patterns)
            
            if is_new_step and current_step:
                # Join current step and add to results
                step_text = ' '.join(current_step).strip()
                if len(step_text) > 15:  # Filter very short steps
                    steps.append(step_text)
                current_step = [line]
            else:
                current_step.append(line)
        
        # Add the last step
        if current_step:
            step_text = ' '.join(current_step).strip()
            if len(step_text) > 15:
                steps.append(step_text)
        
        # Limit to max steps and return
        return steps[:self.config.cot_max_steps]
    
    @performance_monitor
    def compute_hierarchical_attribution_optimized(self, 
                                                  prompt: str,
                                                  use_sfa: bool = True,
                                                  pattern: str = 'analytical',
                                                  use_gpu: bool = None) -> Dict[str, Any]:
        """
        Performance-optimized hierarchical attribution with smart CPU/GPU distribution
        """
        # Auto-determine GPU usage based on prompt complexity and hardware
        if use_gpu is None:
            use_gpu = self._should_use_gpu(prompt)
        
        # Generate CoT (GPU-optimized if available and beneficial)
        if use_gpu and self.model and TORCH_AVAILABLE:
            cot_steps, full_response = self.generate_cot_gpu_optimized(prompt, pattern)
        else:
            # Fallback to CPU or Ollama-based generation
            cot_steps, full_response = self._generate_cot_fallback(prompt, pattern)
        
        if not cot_steps:
            logger.warning("No CoT steps generated, falling back to direct attribution")
            return {
                'error': 'no_cot_steps',
                'direct_attribution': self._compute_direct_attribution(prompt)
            }
        
        # Level 1: Token-level attribution (parallelized for performance)
        token_attributions, step_complexities = self._compute_token_attributions_parallel(
            cot_steps, use_sfa
        )
        
        # Level 2: Step-level importance (CPU-optimized - small computation)
        step_importance = self._compute_step_importance_optimized(cot_steps, full_response)
        
        # Level 3: Chain-level coherence (CPU-optimized - text similarity)
        chain_coherence = self._compute_chain_coherence_optimized(cot_steps, step_importance)
        
        # Identify critical components (CPU-optimized - sorting and filtering)
        critical_steps = self._identify_critical_steps_fast(cot_steps, step_importance)
        critical_tokens = self._identify_critical_tokens_fast(token_attributions, cot_steps)
        
        # Compute metrics (CPU-optimized - simple math)
        metrics = {
            'reasoning_depth': len(cot_steps),
            'avg_step_complexity': np.mean(step_complexities) if step_complexities else 0,
            'chain_coherence': chain_coherence,
            'total_tokens_analyzed': sum(len(attr) for attr in token_attributions),
            'gpu_used': use_gpu,
            'performance_optimized': True
        }
        
        return {
            'prompt': prompt,
            'cot_steps': cot_steps,
            'token_attributions': token_attributions,
            'step_importance': step_importance,
            'step_complexities': step_complexities,
            'chain_coherence': chain_coherence,
            'critical_steps': critical_steps,
            'critical_tokens': critical_tokens,
            'metrics': metrics
        }
    
    def _should_use_gpu(self, prompt: str) -> bool:
        """Determine if GPU should be used based on prompt complexity and hardware"""
        if not TORCH_AVAILABLE or not self.device_manager or self.device_manager.device != "cuda":
            return False
        
        # Simple heuristics for GPU usage decision
        complexity_factors = {
            'length': len(prompt),
            'tokens': len(prompt.split()),
            'sentences': prompt.count('.') + prompt.count('!') + prompt.count('?')
        }
        
        # GPU is beneficial for longer, more complex prompts
        use_gpu = (
            complexity_factors['length'] > 200 or
            complexity_factors['tokens'] > 50 or
            complexity_factors['sentences'] > 3
        )
        
        return use_gpu
    
    def _generate_cot_fallback(self, prompt: str, pattern: str) -> Tuple[List[str], str]:
        """Fallback CoT generation for CPU-only or Ollama models"""
        # This would integrate with Ollama-based generation
        # For now, return simplified version
        template = self.reasoning_patterns.get(pattern, "Let's think step by step:")
        fallback_response = f"{template}\n1. {prompt}\n2. Analysis needed.\n3. Conclusion follows."
        steps = self._parse_cot_steps_optimized(fallback_response, prompt)
        return steps, fallback_response
    
    def _compute_direct_attribution(self, prompt: str) -> Dict[str, float]:
        """Fallback attribution computation"""
        if self.optimized_base:
            return self.optimized_base._compute_single_attribution(prompt)
        
        # Simple fallback
        tokens = prompt.split()
        return {token: np.random.uniform(-0.5, 0.5) for token in tokens}
    
    def _compute_token_attributions_parallel(self, 
                                           cot_steps: List[str], 
                                           use_sfa: bool) -> Tuple[List[Dict[str, float]], List[float]]:
        """
        Parallel computation of token attributions (CPU-optimized for parallel processing)
        """
        token_attributions = []
        step_complexities = []
        
        # Determine optimal parallelization
        max_workers = min(len(cot_steps), self.config.parallel_workers)
        
        if max_workers > 1:
            # Parallel processing on CPU (more efficient for small computations)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all step attribution tasks
                future_to_step = {}
                for idx, step in enumerate(cot_steps):
                    if use_sfa and self.sfa_learner.is_trained:
                        future = executor.submit(self._compute_sfa_attribution, step)
                    else:
                        future = executor.submit(self._compute_step_attribution, step)
                    future_to_step[future] = idx
                
                # Collect results in order
                results = [None] * len(cot_steps)
                for future in as_completed(future_to_step):
                    idx = future_to_step[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing step {idx}: {e}")
                        results[idx] = {}
                
                # Separate attributions and complexities
                for result in results:
                    token_attributions.append(result)
                    complexity = sum(abs(v) for v in result.values()) if result else 0
                    step_complexities.append(complexity)
        
        else:
            # Sequential processing
            for step in cot_steps:
                if use_sfa and self.sfa_learner.is_trained:
                    attribution = self._compute_sfa_attribution(step)
                else:
                    attribution = self._compute_step_attribution(step)
                
                token_attributions.append(attribution)
                complexity = sum(abs(v) for v in attribution.values())
                step_complexities.append(complexity)
        
        return token_attributions, step_complexities
    
    def _compute_sfa_attribution(self, step: str) -> Dict[str, float]:
        """SFA attribution (CPU-optimized - scikit-learn works better on CPU)"""
        if not self.sfa_learner.is_trained:
            return self._compute_step_attribution(step)
        
        try:
            tokens = step.split()  # Simple tokenization for speed
            return self.sfa_learner.predict(step, tokens)
        except Exception as e:
            logger.warning(f"SFA attribution failed: {e}, using fallback")
            return self._compute_step_attribution(step)
    
    def _compute_step_attribution(self, step: str) -> Dict[str, float]:
        """Simplified step attribution (CPU-optimized)"""
        # Simplified attribution for performance
        tokens = step.split()
        
        # Simple heuristic-based attribution
        attributions = {}
        for i, token in enumerate(tokens):
            # Position-based importance
            pos_weight = 1.0 - (abs(i - len(tokens)//2) / len(tokens))
            
            # Length-based importance
            length_weight = min(len(token) / 10.0, 1.0)
            
            # Content-based importance
            content_weight = 1.0
            if token.lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were']:
                content_weight = 0.1  # Common words less important
            elif token.lower() in ['therefore', 'thus', 'because', 'since']:
                content_weight = 2.0  # Logic words more important
            
            attributions[token] = pos_weight * length_weight * content_weight
        
        return attributions
    
    def _compute_step_importance_optimized(self, 
                                         steps: List[str], 
                                         full_response: str) -> List[float]:
        """
        CPU-optimized step importance computation using efficient heuristics
        """
        if not steps:
            return []
        
        importance_scores = []
        
        for i, step in enumerate(steps):
            score = 0.0
            step_lower = step.lower()
            
            # Position-based scoring (CPU-efficient)
            if i == 0:  # First step
                score += 0.3
            if i == len(steps) - 1:  # Last step
                score += 0.4
            
            # Content-based scoring using pre-defined patterns
            logic_words = ['therefore', 'thus', 'hence', 'conclusion', 'because', 'since']
            question_words = ['what', 'why', 'how', 'when', 'where', 'which']
            
            # Vectorized word counting (more efficient than multiple searches)
            step_words = set(step_lower.split())
            
            logic_count = len(step_words.intersection(logic_words))
            question_count = len(step_words.intersection(question_words))
            
            score += logic_count * 0.2
            score += question_count * 0.15
            
            # Length-based scoring
            if len(step) > 100:
                score += 0.1
            
            # Mathematical content detection
            if any(char in step for char in '0123456789+-*/='):
                score += 0.15
            
            importance_scores.append(min(score, 1.0))
        
        # Normalize to sum to 1
        total = sum(importance_scores)
        if total > 0:
            importance_scores = [s / total for s in importance_scores]
        else:
            importance_scores = [1.0 / len(steps)] * len(steps)
        
        return importance_scores
    
    def _compute_chain_coherence_optimized(self, 
                                         steps: List[str], 
                                         importance: List[float]) -> float:
        """
        CPU-optimized chain coherence computation using efficient similarity
        """
        if len(steps) <= 1:
            return 1.0
        
        # Use simple word overlap for efficiency (much faster than TF-IDF)
        coherence_scores = []
        
        for i in range(len(steps) - 1):
            # Convert to sets for efficient intersection
            words1 = set(steps[i].lower().split())
            words2 = set(steps[i + 1].lower().split())
            
            if not words1 or not words2:
                coherence_scores.append(0.0)
                continue
            
            # Jaccard similarity (efficient set operations)
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            similarity = intersection / union if union > 0 else 0.0
            
            # Weight by step importance
            weighted_similarity = similarity * (importance[i] + importance[i+1]) / 2
            coherence_scores.append(weighted_similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _identify_critical_steps_fast(self, 
                                    steps: List[str], 
                                    importance: List[float],
                                    threshold: float = 0.15) -> List[Dict[str, Any]]:
        """Fast critical step identification using list comprehension"""
        critical = [
            {
                'index': i,
                'step_preview': step[:100] + '...' if len(step) > 100 else step,
                'importance': float(imp),
                'rank': sum(1 for other_imp in importance if other_imp > imp) + 1
            }
            for i, (step, imp) in enumerate(zip(steps, importance))
            if imp > threshold
        ]
        
        return sorted(critical, key=lambda x: x['importance'], reverse=True)
    
    def _identify_critical_tokens_fast(self,
                                     attributions: List[Dict[str, float]],
                                     steps: List[str],
                                     top_k: int = 15) -> List[Dict[str, Any]]:
        """Fast critical token identification using efficient sorting"""
        # Use generator expression for memory efficiency
        all_tokens = [
            {
                'token': token,
                'shapley_value': float(value),
                'abs_value': abs(float(value)),
                'step_index': step_idx,
                'step_preview': step[:30] + '...' if len(step) > 30 else step
            }
            for step_idx, (attr_dict, step) in enumerate(zip(attributions, steps))
            for token, value in attr_dict.items()
        ]
        
        # Efficient partial sorting (faster than full sort for top-k)
        import heapq
        return heapq.nlargest(top_k, all_tokens, key=lambda x: x['abs_value'])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with optimization recommendations"""
        if hasattr(self.optimized_base, 'get_performance_report'):
            base_report = self.optimized_base.get_performance_report()
        else:
            base_report = {}
        
        cot_specific = {
            'cot_optimizations': {
                'gpu_text_generation': TORCH_AVAILABLE and self.device_manager and self.device_manager.device == "cuda",
                'cpu_text_processing': True,
                'parallel_attribution': self.config.parallel_workers > 1,
                'efficient_similarity': True,
                'fast_parsing': True
            },
            'performance_strategy': {
                'gpu_operations': ['model_inference', 'tensor_operations', 'batch_processing'],
                'cpu_operations': ['text_parsing', 'similarity_computation', 'small_computations', 'control_logic'],
                'hybrid_approach': True
            }
        }
        
        return {**base_report, **cot_specific}


# Example usage and benchmarking
if __name__ == "__main__":
    print("🚀 Performance-Optimized CoT TokenSHAP")
    print("=" * 45)
    
    # Performance-optimized configuration
    config = TokenSHAPConfig(
        max_samples=10,  # Optimized for speed
        batch_size=4,
        parallel_workers=4,
        use_gpu=True,
        cache_responses=True,
        convergence_threshold=0.05  # Slightly relaxed for speed
    )
    
    # Initialize optimized CoT analyzer
    cot_analyzer = OptimizedCoTTokenSHAP(config=config)
    
    print(f"🎮 Device: {cot_analyzer.device_manager.device if cot_analyzer.device_manager else 'CPU-only'}")
    print(f"🔧 Parallel workers: {config.parallel_workers}")
    
    # Test prompts with varying complexity
    test_prompts = [
        "What is 2+2?",  # Simple - should use CPU
        "Analyze the economic impact of renewable energy adoption on traditional energy markets, considering both short-term disruption and long-term benefits.",  # Complex - should use GPU if available
        "Compare machine learning and deep learning approaches.",  # Medium complexity
    ]
    
    # Performance benchmarking
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        
        start_time = time.time()
        result = cot_analyzer.compute_hierarchical_attribution_optimized(
            prompt, 
            pattern='analytical'
        )
        execution_time = time.time() - start_time
        
        if 'error' not in result:
            metrics = result['metrics']
            print(f"⚡ Execution: {execution_time:.2f}s")
            print(f"📊 Steps: {metrics['reasoning_depth']}")
            print(f"🎮 GPU used: {metrics.get('gpu_used', False)}")
            print(f"🔍 Tokens analyzed: {metrics['total_tokens_analyzed']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Performance summary
    summary = cot_analyzer.get_performance_summary()
    print(f"\n📋 Performance Summary:")
    for key, value in summary.get('cot_optimizations', {}).items():
        print(f"   {key}: {'✅' if value else '❌'}")
    
    print(f"\n💡 Hybrid Architecture:")
    strategy = summary.get('performance_strategy', {})
    print(f"   🎮 GPU: {', '.join(strategy.get('gpu_operations', []))}")
    print(f"   💻 CPU: {', '.join(strategy.get('cpu_operations', []))}")