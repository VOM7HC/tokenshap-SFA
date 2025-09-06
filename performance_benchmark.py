"""
Comprehensive Performance Benchmark for Optimized TokenSHAP
"""

import sys
sys.path.append('.')

import time
import psutil
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from config import TokenSHAPConfig

# Try to import optimized versions
try:
    from optimized_performance_manager import OptimizedTokenSHAP, DeviceManager
    from cot_explainer_optimized import OptimizedCoTTokenSHAP
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

# Fallback imports
try:
    from tokenshap_ollama import TokenSHAPWithOllama
    from cot_ollama_reasoning import OllamaCoTAnalyzer
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    method_name: str
    execution_time: float
    memory_usage_mb: float
    throughput_ops_sec: float
    device_used: str
    success: bool
    error_message: str = ""
    additional_metrics: Dict[str, Any] = None


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self):
        self.results = []
        self.system_info = self._get_system_info()
        print("🔍 Performance Benchmark Suite")
        print("=" * 40)
        self._print_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context"""
        info = {
            'cpu_cores': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total_gb': psutil.virtual_memory().total / 1e9,
            'memory_available_gb': psutil.virtual_memory().available / 1e9
        }
        
        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                info['gpu_available'] = True
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            else:
                info['gpu_available'] = False
        except ImportError:
            info['gpu_available'] = False
        
        return info
    
    def _print_system_info(self):
        """Print system information"""
        print(f"💻 System Information:")
        print(f"   CPU Cores: {self.system_info['cpu_cores']}")
        print(f"   CPU Frequency: {self.system_info['cpu_freq']:.0f} MHz")
        print(f"   Total Memory: {self.system_info['memory_total_gb']:.1f} GB")
        print(f"   Available Memory: {self.system_info['memory_available_gb']:.1f} GB")
        
        if self.system_info['gpu_available']:
            print(f"   🎮 GPU: {self.system_info['gpu_name']}")
            print(f"   GPU Memory: {self.system_info['gpu_memory_gb']:.1f} GB")
        else:
            print(f"   🎮 GPU: Not available")
        print()
    
    def _measure_performance(self, func, *args, **kwargs) -> BenchmarkResult:
        """Measure performance of a function"""
        
        # Initial memory measurement
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1e6  # MB
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_message = ""
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1e6  # MB
        memory_usage = final_memory - initial_memory
        
        # Calculate throughput
        throughput = 1.0 / execution_time if execution_time > 0 else 0
        
        return BenchmarkResult(
            method_name=func.__name__ if hasattr(func, '__name__') else str(func),
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            throughput_ops_sec=throughput,
            device_used="unknown",  # Will be updated by specific benchmarks
            success=success,
            error_message=error_message
        )
    
    def benchmark_optimized_tokenshap(self, test_prompts: List[str]) -> List[BenchmarkResult]:
        """Benchmark optimized TokenSHAP implementation"""
        results = []
        
        if not OPTIMIZED_AVAILABLE:
            print("⚠️  Optimized TokenSHAP not available - skipping")
            return results
        
        print("🚀 Benchmarking Optimized TokenSHAP...")
        
        try:
            # Test different configurations
            configs = [
                ("CPU-Optimized", TokenSHAPConfig(use_gpu=False, parallel_workers=4, max_samples=5)),
                ("GPU-Optimized", TokenSHAPConfig(use_gpu=True, parallel_workers=2, max_samples=10)),
                ("Hybrid", TokenSHAPConfig(use_gpu=True, parallel_workers=4, max_samples=8))
            ]
            
            for config_name, config in configs:
                print(f"   📊 Testing {config_name} configuration...")
                
                try:
                    optimizer = OptimizedTokenSHAP(config=config)
                    
                    # Single prompt benchmark
                    result = self._measure_performance(
                        optimizer.adaptive_compute_strategy,
                        test_prompts[0]
                    )
                    result.method_name = f"OptimizedTokenSHAP-{config_name}-Single"
                    result.device_used = optimizer.device_manager.device
                    results.append(result)
                    
                    # Batch benchmark
                    if len(test_prompts) > 1:
                        result = self._measure_performance(
                            optimizer.optimized_batch_attribution,
                            test_prompts[:3]  # Test with first 3 prompts
                        )
                        result.method_name = f"OptimizedTokenSHAP-{config_name}-Batch"
                        result.device_used = optimizer.device_manager.device
                        result.throughput_ops_sec *= 3  # Adjust for batch size
                        results.append(result)
                    
                except Exception as e:
                    print(f"   ❌ {config_name} configuration failed: {e}")
        
        except Exception as e:
            print(f"   ❌ Optimized TokenSHAP benchmark failed: {e}")
        
        return results
    
    def benchmark_optimized_cot(self, test_prompts: List[str]) -> List[BenchmarkResult]:
        """Benchmark optimized CoT implementation"""
        results = []
        
        if not OPTIMIZED_AVAILABLE:
            print("⚠️  Optimized CoT not available - skipping")
            return results
        
        print("🧠 Benchmarking Optimized CoT...")
        
        try:
            config = TokenSHAPConfig(
                max_samples=5,  # Fast for benchmarking
                parallel_workers=4,
                use_gpu=True
            )
            
            cot_analyzer = OptimizedCoTTokenSHAP(config=config)
            
            for i, prompt in enumerate(test_prompts[:2]):  # Test first 2 prompts
                result = self._measure_performance(
                    cot_analyzer.compute_hierarchical_attribution_optimized,
                    prompt,
                    use_sfa=False,  # Test without SFA first
                    pattern='analytical'
                )
                result.method_name = f"OptimizedCoT-Prompt{i+1}"
                result.device_used = cot_analyzer.device_manager.device if cot_analyzer.device_manager else "cpu"
                results.append(result)
        
        except Exception as e:
            print(f"   ❌ Optimized CoT benchmark failed: {e}")
        
        return results
    
    def benchmark_ollama_implementations(self, test_prompts: List[str]) -> List[BenchmarkResult]:
        """Benchmark Ollama-based implementations"""
        results = []
        
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama implementations not available - skipping")
            return results
        
        print("🦙 Benchmarking Ollama Implementations...")
        
        # Test if Ollama server is available
        try:
            import requests
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("   ⚠️  Ollama server not accessible - skipping Ollama benchmarks")
                return results
        except:
            print("   ⚠️  Ollama server not accessible - skipping Ollama benchmarks")
            return results
        
        try:
            # Check available models
            models_response = requests.get("http://127.0.0.1:11434/api/tags").json()
            available_models = [model['name'] for model in models_response.get('models', [])]
            
            if not available_models:
                print("   ⚠️  No Ollama models available")
                return results
            
            test_model = available_models[0]  # Use first available model
            print(f"   📋 Using model: {test_model}")
            
            # Benchmark TokenSHAPWithOllama
            config = TokenSHAPConfig(max_samples=3)  # Very fast for benchmarking
            ollama_explainer = TokenSHAPWithOllama(
                model_name=test_model,
                config=config
            )
            
            result = self._measure_performance(
                ollama_explainer.explain,
                test_prompts[0]
            )
            result.method_name = "TokenSHAPWithOllama"
            result.device_used = "cpu-ollama"
            results.append(result)
            
            # Benchmark OllamaCoTAnalyzer
            cot_analyzer = OllamaCoTAnalyzer(
                model_name=test_model,
                config=config
            )
            
            result = self._measure_performance(
                cot_analyzer.analyze_cot_attribution,
                test_prompts[0],
                analyze_steps=False  # Skip token analysis for speed
            )
            result.method_name = "OllamaCoTAnalyzer"
            result.device_used = "cpu-ollama"
            results.append(result)
        
        except Exception as e:
            print(f"   ❌ Ollama benchmark failed: {e}")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        
        # Test prompts with varying complexity
        test_prompts = [
            "What is machine learning?",
            "Explain the differences between supervised and unsupervised learning algorithms, including their use cases and performance characteristics.",
            "How does deep learning work?",
            "Analyze the economic and environmental impact of artificial intelligence adoption in manufacturing industries, considering both benefits and challenges."
        ]
        
        print(f"📝 Test prompts: {len(test_prompts)}")
        for i, prompt in enumerate(test_prompts, 1):
            preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"   {i}. {preview}")
        print()
        
        # Run all benchmarks
        all_results = []
        
        # 1. Optimized implementations
        all_results.extend(self.benchmark_optimized_tokenshap(test_prompts))
        all_results.extend(self.benchmark_optimized_cot(test_prompts))
        
        # 2. Ollama implementations
        all_results.extend(self.benchmark_ollama_implementations(test_prompts))
        
        self.results = all_results
        
        # Generate comprehensive report
        return self._generate_comprehensive_report()
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Separate successful and failed results
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        if not successful:
            return {"error": "All benchmarks failed", "failed_results": failed}
        
        # Performance analysis
        performance_analysis = {
            'fastest_method': min(successful, key=lambda x: x.execution_time),
            'slowest_method': max(successful, key=lambda x: x.execution_time),
            'most_memory_efficient': min(successful, key=lambda x: x.memory_usage_mb),
            'highest_throughput': max(successful, key=lambda x: x.throughput_ops_sec)
        }
        
        # Device utilization
        device_stats = {}
        for result in successful:
            device = result.device_used
            if device not in device_stats:
                device_stats[device] = {'count': 0, 'avg_time': 0, 'total_time': 0}
            device_stats[device]['count'] += 1
            device_stats[device]['total_time'] += result.execution_time
        
        for device in device_stats:
            device_stats[device]['avg_time'] = device_stats[device]['total_time'] / device_stats[device]['count']
        
        # Performance recommendations
        recommendations = self._generate_recommendations(successful, device_stats)
        
        report = {
            'system_info': self.system_info,
            'total_benchmarks': len(self.results),
            'successful_benchmarks': len(successful),
            'failed_benchmarks': len(failed),
            'performance_analysis': {
                'fastest': {
                    'method': performance_analysis['fastest_method'].method_name,
                    'time': performance_analysis['fastest_method'].execution_time,
                    'device': performance_analysis['fastest_method'].device_used
                },
                'most_efficient_memory': {
                    'method': performance_analysis['most_memory_efficient'].method_name,
                    'memory_mb': performance_analysis['most_memory_efficient'].memory_usage_mb,
                    'device': performance_analysis['most_memory_efficient'].device_used
                },
                'highest_throughput': {
                    'method': performance_analysis['highest_throughput'].method_name,
                    'throughput': performance_analysis['highest_throughput'].throughput_ops_sec,
                    'device': performance_analysis['highest_throughput'].device_used
                }
            },
            'device_utilization': device_stats,
            'recommendations': recommendations,
            'detailed_results': [
                {
                    'method': r.method_name,
                    'time_sec': r.execution_time,
                    'memory_mb': r.memory_usage_mb,
                    'throughput_ops_sec': r.throughput_ops_sec,
                    'device': r.device_used,
                    'success': r.success
                }
                for r in successful
            ]
        }
        
        if failed:
            report['failed_methods'] = [
                {'method': r.method_name, 'error': r.error_message}
                for r in failed
            ]
        
        return report
    
    def _generate_recommendations(self, 
                                successful_results: List[BenchmarkResult],
                                device_stats: Dict[str, Dict]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # GPU utilization recommendations
        gpu_results = [r for r in successful_results if 'cuda' in r.device_used.lower()]
        cpu_results = [r for r in successful_results if 'cpu' in r.device_used.lower()]
        
        if gpu_results and cpu_results:
            avg_gpu_time = np.mean([r.execution_time for r in gpu_results])
            avg_cpu_time = np.mean([r.execution_time for r in cpu_results])
            
            if avg_gpu_time < avg_cpu_time * 0.8:
                recommendations.append("GPU acceleration provides significant performance benefits - consider using GPU for complex operations")
            elif avg_cpu_time < avg_gpu_time * 0.8:
                recommendations.append("CPU implementation is faster for your workload - GPU overhead may not be worth it")
            else:
                recommendations.append("GPU and CPU performance are similar - use hybrid approach based on task complexity")
        
        # Memory usage recommendations
        high_memory_methods = [r for r in successful_results if r.memory_usage_mb > 100]
        if high_memory_methods:
            recommendations.append("High memory usage detected in some methods - consider reducing batch size or using streaming")
        
        # Throughput recommendations
        low_throughput_methods = [r for r in successful_results if r.throughput_ops_sec < 0.1]
        if low_throughput_methods:
            recommendations.append("Low throughput detected - consider parallel processing or batch optimization")
        
        # Model-specific recommendations
        ollama_results = [r for r in successful_results if 'ollama' in r.method_name.lower()]
        if ollama_results:
            avg_ollama_time = np.mean([r.execution_time for r in ollama_results])
            if avg_ollama_time > 10:  # More than 10 seconds
                recommendations.append("Ollama operations are slow - consider using smaller models or reducing max_samples")
        
        return recommendations or ["Performance looks good - no specific recommendations"]
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted benchmark report"""
        
        print("\n📊 COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 50)
        
        # System info summary
        print(f"🖥️  System: {report['system_info']['cpu_cores']} cores, "
              f"{report['system_info']['memory_total_gb']:.1f}GB RAM, "
              f"GPU: {'Yes' if report['system_info']['gpu_available'] else 'No'}")
        
        # Benchmark summary
        print(f"📈 Benchmarks: {report['successful_benchmarks']}/{report['total_benchmarks']} successful")
        
        if 'performance_analysis' in report:
            analysis = report['performance_analysis']
            print(f"\n🏆 Performance Winners:")
            print(f"   ⚡ Fastest: {analysis['fastest']['method']} ({analysis['fastest']['time']:.2f}s)")
            print(f"   💾 Most Memory Efficient: {analysis['most_efficient_memory']['method']} ({analysis['most_efficient_memory']['memory_mb']:.1f}MB)")
            print(f"   🚀 Highest Throughput: {analysis['highest_throughput']['method']} ({analysis['highest_throughput']['throughput']:.2f} ops/sec)")
        
        # Device utilization
        if 'device_utilization' in report:
            print(f"\n🎮 Device Utilization:")
            for device, stats in report['device_utilization'].items():
                print(f"   {device}: {stats['count']} operations, avg {stats['avg_time']:.2f}s")
        
        # Detailed results
        if 'detailed_results' in report:
            print(f"\n📋 Detailed Results:")
            print(f"{'Method':<30} {'Time(s)':<8} {'Memory(MB)':<12} {'Throughput':<12} {'Device':<15}")
            print("-" * 80)
            
            for result in sorted(report['detailed_results'], key=lambda x: x['time_sec']):
                print(f"{result['method']:<30} {result['time_sec']:<8.2f} "
                      f"{result['memory_mb']:<12.1f} {result['throughput_ops_sec']:<12.2f} "
                      f"{result['device']:<15}")
        
        # Recommendations
        if 'recommendations' in report:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Failed methods
        if 'failed_methods' in report:
            print(f"\n❌ Failed Methods:")
            for failed in report['failed_methods']:
                print(f"   {failed['method']}: {failed['error']}")
        
        print("\n" + "=" * 50)


def main():
    """Main benchmark execution"""
    
    benchmark = PerformanceBenchmark()
    
    # Run comprehensive benchmark
    report = benchmark.run_comprehensive_benchmark()
    
    # Print detailed report
    benchmark.print_report(report)
    
    # Save report to file
    import json
    try:
        with open('performance_benchmark_report.json', 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Clean the report for JSON serialization
            clean_report = json.loads(json.dumps(report, default=convert_numpy))
            json.dump(clean_report, f, indent=2)
        
        print(f"📄 Report saved to: performance_benchmark_report.json")
    
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")


if __name__ == "__main__":
    main()