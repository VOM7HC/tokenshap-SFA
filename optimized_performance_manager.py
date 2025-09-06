"""
Performance-Optimized Manager for TokenSHAP - Hybrid CPU/GPU Architecture
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, PreTrainedModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create placeholder types for when transformers is not available
    class PreTrainedModel:
        pass
    class AutoTokenizer:
        pass

from config import TokenSHAPConfig, AttributionMethod

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track performance metrics for optimization"""
    gpu_time: float = 0.0
    cpu_time: float = 0.0
    memory_usage: float = 0.0
    batch_size: int = 0
    throughput: float = 0.0
    device_used: str = "unknown"


class DeviceManager:
    """Smart device management for optimal CPU/GPU usage"""
    
    def __init__(self, config: TokenSHAPConfig):
        self.config = config
        self.device = self._select_optimal_device()
        self.device_capabilities = self._analyze_device_capabilities()
        logger.info(f"Device Manager initialized: {self.device}")
    
    def _select_optimal_device(self) -> str:
        """Select the optimal device based on availability and workload"""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if self.config.use_gpu and torch.cuda.is_available():
            # Check GPU memory
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory > 4:  # At least 4GB GPU memory
                    return "cuda"
                else:
                    logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient, using CPU")
                    return "cpu"
            except:
                return "cpu"
        
        return "cpu"
    
    def _analyze_device_capabilities(self) -> Dict[str, Any]:
        """Analyze device capabilities for optimization"""
        capabilities = {"device": self.device}
        
        if self.device == "cuda" and TORCH_AVAILABLE:
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                capabilities.update({
                    "name": gpu_props.name,
                    "memory_total": gpu_props.total_memory / 1e9,
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multiprocessors": gpu_props.multi_processor_count
                })
            except Exception as e:
                logger.warning(f"Could not analyze GPU capabilities: {e}")
        
        return capabilities
    
    def to_device(self, tensor_or_dict):
        """Move tensors to optimal device"""
        if not TORCH_AVAILABLE or self.device == "cpu":
            return tensor_or_dict
        
        if isinstance(tensor_or_dict, dict):
            return {k: v.to(self.device) if hasattr(v, 'to') else v 
                   for k, v in tensor_or_dict.items()}
        elif hasattr(tensor_or_dict, 'to'):
            return tensor_or_dict.to(self.device)
        
        return tensor_or_dict


def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        # Track GPU memory if available
        if hasattr(self, 'device_manager') and self.device_manager.device == "cuda" and TORCH_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
        
        result = func(self, *args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate memory usage
        if hasattr(self, 'device_manager') and self.device_manager.device == "cuda" and TORCH_AVAILABLE:
            peak_memory = torch.cuda.max_memory_allocated() - start_memory
            memory_mb = peak_memory / 1e6
        else:
            memory_mb = 0
        
        # Store metrics
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = []
        
        metrics = PerformanceMetrics(
            gpu_time=execution_time if hasattr(self, 'device_manager') and self.device_manager.device == "cuda" else 0,
            cpu_time=execution_time if not hasattr(self, 'device_manager') or self.device_manager.device == "cpu" else 0,
            memory_usage=memory_mb,
            device_used=getattr(self.device_manager, 'device', 'cpu') if hasattr(self, 'device_manager') else 'cpu',
            throughput=1.0 / execution_time if execution_time > 0 else 0
        )
        
        self.performance_metrics.append(metrics)
        
        logger.debug(f"{func.__name__}: {execution_time:.3f}s, {memory_mb:.1f}MB, {metrics.device_used}")
        
        return result
    
    return wrapper


class OptimizedTokenSHAP:
    """Performance-optimized TokenSHAP with intelligent CPU/GPU distribution"""
    
    def __init__(self, 
                 model: Optional[PreTrainedModel] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 config: TokenSHAPConfig = None):
        
        self.config = config or TokenSHAPConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.performance_metrics = []
        
        # Initialize device manager
        self.device_manager = DeviceManager(config)
        
        # Move model to optimal device
        if self.model and TORCH_AVAILABLE:
            self.model = self.device_manager.to_device(self.model)
        
        # Initialize performance-optimized components
        self._init_optimized_components()
        
        logger.info("OptimizedTokenSHAP initialized with hybrid CPU/GPU architecture")
    
    def _init_optimized_components(self):
        """Initialize components optimized for their best execution environment"""
        
        # GPU-optimized: Model inference, tensor operations
        self.gpu_operations = {
            'model_inference': self.device_manager.device == "cuda",
            'tensor_math': self.device_manager.device == "cuda",
            'batch_processing': self.device_manager.device == "cuda"
        }
        
        # CPU-optimized: Text processing, small computations, control logic
        self.cpu_operations = {
            'tokenization': True,  # Often faster on CPU
            'text_parsing': True,
            'small_arrays': True,  # NumPy operations < 1000 elements
            'control_logic': True
        }
        
        logger.info(f"Performance distribution: GPU={sum(self.gpu_operations.values())} ops, CPU={sum(self.cpu_operations.values())} ops")
    
    @performance_monitor
    def optimized_model_inference(self, inputs: Dict[str, Any], **generation_kwargs) -> Any:
        """GPU-optimized model inference with smart batching"""
        if not self.model or not TORCH_AVAILABLE:
            raise ValueError("Model and PyTorch required for optimized inference")
        
        # Move inputs to optimal device
        inputs = self.device_manager.to_device(inputs)
        
        # Use mixed precision if available on GPU
        if self.device_manager.device == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwargs)
        else:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
        
        return outputs
    
    @performance_monitor
    def optimized_batch_attribution(self, 
                                   prompts: List[str], 
                                   batch_size: Optional[int] = None) -> List[Dict[str, float]]:
        """Batch process multiple prompts with optimal GPU utilization"""
        
        if not batch_size:
            # Auto-determine optimal batch size based on device
            if self.device_manager.device == "cuda":
                gpu_memory_gb = self.device_manager.device_capabilities.get("memory_total", 4)
                batch_size = min(32, max(2, int(gpu_memory_gb * 2)))  # Heuristic: 2 samples per GB
            else:
                batch_size = min(8, max(1, self.config.parallel_workers))
        
        results = []
        
        # Process in optimized batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            if self.device_manager.device == "cuda" and len(batch) > 1:
                # GPU: Process batch simultaneously
                batch_results = self._gpu_batch_process(batch)
            else:
                # CPU: Process in parallel threads
                batch_results = self._cpu_parallel_process(batch)
            
            results.extend(batch_results)
        
        return results
    
    def _gpu_batch_process(self, batch: List[str]) -> List[Dict[str, float]]:
        """GPU-optimized batch processing"""
        if not TORCH_AVAILABLE or not self.model:
            return self._cpu_parallel_process(batch)
        
        try:
            # Tokenize entire batch
            if self.tokenizer:
                batch_inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=self.config.max_input_length
                )
                batch_inputs = self.device_manager.to_device(batch_inputs)
            else:
                # Fallback to CPU processing
                return self._cpu_parallel_process(batch)
            
            # Generate responses in batch
            with torch.no_grad():
                if self.device_manager.device == "cuda":
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model.generate(
                            **batch_inputs,
                            max_length=self.config.max_output_length,
                            temperature=self.config.temperature,
                            do_sample=True,
                            top_p=0.95,
                            num_return_sequences=1
                        )
                else:
                    outputs = self.model.generate(
                        **batch_inputs,
                        max_length=self.config.max_output_length,
                        temperature=self.config.temperature,
                        do_sample=True,
                        top_p=0.95,
                        num_return_sequences=1
                    )
            
            # Process results (on CPU for efficiency)
            results = []
            for i, prompt in enumerate(batch):
                # Extract individual result and compute attribution
                individual_output = outputs[i:i+1]
                response = self.tokenizer.decode(individual_output[0], skip_special_tokens=True)
                
                # Compute Shapley values (this stays CPU-optimized for now)
                attribution = self._compute_attribution_cpu(prompt, response)
                results.append(attribution)
            
            return results
        
        except Exception as e:
            logger.warning(f"GPU batch processing failed: {e}, falling back to CPU")
            return self._cpu_parallel_process(batch)
    
    def _cpu_parallel_process(self, batch: List[str]) -> List[Dict[str, float]]:
        """CPU-optimized parallel processing"""
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(self._compute_single_attribution, prompt): prompt 
                for prompt in batch
            }
            
            results = []
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing prompt '{prompt[:50]}...': {e}")
                    results.append({})
        
        return results
    
    def _compute_single_attribution(self, prompt: str) -> Dict[str, float]:
        """Compute attribution for a single prompt - optimized for current hardware"""
        # This is a simplified placeholder - would integrate with actual TokenSHAP logic
        # The key is to keep small computations on CPU
        
        if not self.tokenizer:
            # Simple word-based tokenization for CPU
            tokens = prompt.lower().split()
        else:
            tokens = self.tokenizer.tokenize(prompt)
        
        # Simulate Shapley value computation (CPU-optimized)
        attributions = {}
        for token in tokens:
            # Placeholder: real implementation would compute actual Shapley values
            attributions[token] = np.random.uniform(-1, 1)  # CPU-efficient
        
        return attributions
    
    def _compute_attribution_cpu(self, prompt: str, response: str) -> Dict[str, float]:
        """CPU-optimized attribution computation"""
        # Keep this on CPU as it involves a lot of small, irregular computations
        return self._compute_single_attribution(prompt)
    
    @performance_monitor
    def adaptive_compute_strategy(self, 
                                 prompt: str, 
                                 complexity_threshold: float = 0.5) -> Dict[str, float]:
        """Adaptively choose computation strategy based on prompt complexity"""
        
        # Analyze prompt complexity (CPU operation)
        complexity = self._analyze_prompt_complexity(prompt)
        
        if complexity > complexity_threshold and self.device_manager.device == "cuda":
            # High complexity -> Use GPU acceleration
            return self._gpu_accelerated_attribution(prompt)
        else:
            # Low complexity -> CPU is more efficient
            return self._cpu_optimized_attribution(prompt)
    
    def _analyze_prompt_complexity(self, prompt: str) -> float:
        """Analyze prompt complexity to determine optimal processing strategy"""
        factors = {
            'length': len(prompt) / 1000.0,  # Normalize by typical length
            'tokens': len(prompt.split()) / 100.0,  # Token count factor
            'sentences': prompt.count('.') / 10.0,  # Sentence complexity
            'special_chars': sum(1 for c in prompt if not c.isalnum() and c != ' ') / len(prompt) if prompt else 0
        }
        
        # Weighted complexity score
        complexity = (
            factors['length'] * 0.4 +
            factors['tokens'] * 0.3 +
            factors['sentences'] * 0.2 +
            factors['special_chars'] * 0.1
        )
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    def _gpu_accelerated_attribution(self, prompt: str) -> Dict[str, float]:
        """GPU-accelerated attribution for complex prompts"""
        if not TORCH_AVAILABLE or self.device_manager.device != "cuda":
            return self._cpu_optimized_attribution(prompt)
        
        # GPU-optimized implementation would go here
        # For now, fall back to CPU
        return self._cpu_optimized_attribution(prompt)
    
    def _cpu_optimized_attribution(self, prompt: str) -> Dict[str, float]:
        """CPU-optimized attribution for simple prompts"""
        return self._compute_single_attribution(prompt)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_metrics:
            return {"error": "No performance data available"}
        
        # Aggregate metrics
        total_gpu_time = sum(m.gpu_time for m in self.performance_metrics)
        total_cpu_time = sum(m.cpu_time for m in self.performance_metrics)
        avg_memory = np.mean([m.memory_usage for m in self.performance_metrics])
        avg_throughput = np.mean([m.throughput for m in self.performance_metrics])
        
        device_usage = {}
        for metric in self.performance_metrics:
            device_usage[metric.device_used] = device_usage.get(metric.device_used, 0) + 1
        
        return {
            'total_operations': len(self.performance_metrics),
            'gpu_time_total': total_gpu_time,
            'cpu_time_total': total_cpu_time,
            'gpu_cpu_ratio': total_gpu_time / total_cpu_time if total_cpu_time > 0 else float('inf'),
            'avg_memory_usage_mb': avg_memory,
            'avg_throughput_ops_sec': avg_throughput,
            'device_distribution': device_usage,
            'device_capabilities': self.device_manager.device_capabilities,
            'optimization_recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []
        
        if not self.performance_metrics:
            return ["Run some operations first to generate recommendations"]
        
        # Analyze GPU vs CPU usage
        gpu_ops = sum(1 for m in self.performance_metrics if m.device_used == "cuda")
        cpu_ops = len(self.performance_metrics) - gpu_ops
        
        if gpu_ops == 0 and self.device_manager.device == "cuda":
            recommendations.append("GPU available but not used - consider enabling GPU operations")
        
        # Memory usage analysis
        avg_memory = np.mean([m.memory_usage for m in self.performance_metrics])
        if avg_memory > 1000:  # > 1GB
            recommendations.append("High memory usage detected - consider reducing batch size")
        
        # Throughput analysis
        avg_throughput = np.mean([m.throughput for m in self.performance_metrics])
        if avg_throughput < 0.1:  # Less than 0.1 ops/sec
            recommendations.append("Low throughput detected - consider parallel processing")
        
        return recommendations or ["Performance looks good - no specific recommendations"]


# Example usage and integration
if __name__ == "__main__":
    print("🚀 Performance-Optimized TokenSHAP")
    print("=" * 40)
    
    # Example configuration for performance
    config = TokenSHAPConfig(
        max_samples=20,
        batch_size=8,
        parallel_workers=4,
        use_gpu=True,  # Will auto-detect and fallback if needed
        cache_responses=True
    )
    
    # Initialize optimized system
    optimizer = OptimizedTokenSHAP(config=config)
    
    # Example performance test
    test_prompts = [
        "Simple prompt for testing",
        "More complex prompt with multiple sentences. This has various complexity factors.",
        "Short",
        "A very long prompt that contains multiple ideas, several sentences, and various punctuation marks! It should trigger the GPU acceleration path due to its complexity."
    ]
    
    print(f"📊 Testing with {len(test_prompts)} prompts")
    print(f"🎮 Device: {optimizer.device_manager.device}")
    print(f"💾 Device capabilities: {optimizer.device_manager.device_capabilities}")
    
    # Test batch processing
    start_time = time.time()
    results = optimizer.optimized_batch_attribution(test_prompts)
    total_time = time.time() - start_time
    
    print(f"⚡ Batch processing completed in {total_time:.2f}s")
    print(f"📈 Throughput: {len(test_prompts)/total_time:.2f} prompts/sec")
    
    # Generate performance report
    report = optimizer.get_performance_report()
    print(f"\n📋 Performance Report:")
    for key, value in report.items():
        if key != 'optimization_recommendations':
            print(f"   {key}: {value}")
    
    print(f"\n💡 Recommendations:")
    for rec in report['optimization_recommendations']:
        print(f"   • {rec}")