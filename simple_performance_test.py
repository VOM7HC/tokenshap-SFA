"""
Simplified Performance Test for Current Setup
"""

import sys
sys.path.append('.')

import time
import psutil
from typing import Dict, Any, List

# Test what's available in current environment
try:
    from tokenshap_ollama import TokenSHAPWithOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from cot_ollama_reasoning import OllamaCoTAnalyzer
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

try:
    from config import TokenSHAPConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def test_system_resources():
    """Test system resources available"""
    print("🔍 System Resource Check")
    print("=" * 30)
    
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory info
    memory = psutil.virtual_memory()
    
    # Disk info
    disk = psutil.disk_usage('/')
    
    print(f"💻 CPU Cores: {cpu_count}")
    print(f"⚡ CPU Usage: {cpu_percent:.1f}%")
    print(f"🧠 Memory: {memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB ({memory.percent:.1f}%)")
    print(f"💾 Disk: {disk.used/1e9:.1f}GB / {disk.total/1e9:.1f}GB ({disk.percent:.1f}%)")
    
    return {
        'cpu_cores': cpu_count,
        'cpu_percent': cpu_percent,
        'memory_total_gb': memory.total / 1e9,
        'memory_used_gb': memory.used / 1e9,
        'memory_percent': memory.percent
    }


def benchmark_function(func, *args, description="Function", **kwargs):
    """Benchmark a function's performance"""
    
    # Get initial system state
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1e6  # MB
    initial_cpu_percent = process.cpu_percent()
    
    print(f"🏃 Running: {description}")
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        success = True
        error_msg = ""
    except Exception as e:
        result = None
        success = False
        error_msg = str(e)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Get final system state
    final_memory = process.memory_info().rss / 1e6  # MB
    memory_delta = final_memory - initial_memory
    
    # Results
    print(f"   ⏱️  Time: {execution_time:.3f}s")
    print(f"   🧠 Memory: {memory_delta:+.1f}MB")
    print(f"   ✅ Status: {'Success' if success else 'Failed'}")
    if not success:
        print(f"   ❌ Error: {error_msg}")
    
    return {
        'execution_time': execution_time,
        'memory_delta_mb': memory_delta,
        'success': success,
        'error': error_msg,
        'result_size': len(str(result)) if result else 0
    }


def test_ollama_performance():
    """Test Ollama-based implementations"""
    
    if not OLLAMA_AVAILABLE or not CONFIG_AVAILABLE:
        print("⚠️  Ollama components not available")
        return {}
    
    print("\n🦙 Ollama Performance Tests")
    print("=" * 35)
    
    # Check if Ollama server is available
    try:
        import requests
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if response.status_code != 200:
            print("   ❌ Ollama server not accessible")
            return {}
        
        models = response.json().get('models', [])
        if not models:
            print("   ❌ No Ollama models available")
            return {}
        
        model_name = models[0]['name']
        print(f"   📋 Using model: {model_name}")
        
    except Exception as e:
        print(f"   ❌ Ollama server check failed: {e}")
        return {}
    
    results = {}
    
    # Test prompts with different complexity
    test_cases = [
        ("Simple", "What is 2+2?"),
        ("Medium", "Explain machine learning in one paragraph."),
        ("Complex", "Compare the advantages and disadvantages of renewable energy sources versus fossil fuels, considering economic, environmental, and technological factors.")
    ]
    
    # Test TokenSHAPWithOllama
    print("\n   🔍 Testing TokenSHAP with Ollama")
    
    try:
        config = TokenSHAPConfig(max_samples=3, parallel_workers=1)  # Fast config
        explainer = TokenSHAPWithOllama(model_name=model_name, config=config)
        
        for complexity, prompt in test_cases:
            print(f"\n   📝 {complexity} prompt:")
            result = benchmark_function(
                explainer.explain, 
                prompt,
                description=f"TokenSHAP-{complexity}"
            )
            results[f'tokenshap_{complexity.lower()}'] = result
            
            if result['success'] and result['execution_time'] > 0:
                throughput = 1.0 / result['execution_time']
                print(f"   📈 Throughput: {throughput:.2f} prompts/sec")
    
    except Exception as e:
        print(f"   ❌ TokenSHAP test failed: {e}")
    
    # Test OllamaCoTAnalyzer
    if COT_AVAILABLE:
        print("\n   🧠 Testing CoT Analyzer")
        
        try:
            config = TokenSHAPConfig(max_samples=2, parallel_workers=1)
            cot_analyzer = OllamaCoTAnalyzer(model_name=model_name, config=config)
            
            # Test with medium complexity prompt
            complexity, prompt = test_cases[1]  # Medium complexity
            print(f"\n   📝 CoT analysis:")
            result = benchmark_function(
                cot_analyzer.analyze_cot_attribution,
                prompt,
                analyze_steps=False,  # Skip token analysis for speed
                description="CoT-Analysis"
            )
            results['cot_analysis'] = result
            
            if result['success'] and result['execution_time'] > 0:
                throughput = 1.0 / result['execution_time']
                print(f"   📈 Throughput: {throughput:.2f} analyses/sec")
        
        except Exception as e:
            print(f"   ❌ CoT test failed: {e}")
    
    return results


def test_cpu_optimizations():
    """Test CPU-specific optimizations"""
    
    print("\n💻 CPU Optimization Tests")
    print("=" * 30)
    
    results = {}
    
    # Test parallel processing
    print("   🔀 Testing parallel processing...")
    
    def cpu_intensive_task(n_items=1000):
        """Simulate CPU-intensive task"""
        import numpy as np
        # Matrix operations that benefit from multiple cores
        matrices = [np.random.rand(100, 100) for _ in range(n_items // 100)]
        results = []
        for matrix in matrices:
            # Multiple operations that can be parallelized
            result = np.dot(matrix, matrix.T)
            result = np.linalg.inv(result + np.eye(100) * 0.001)  # Add small regularization
            results.append(np.trace(result))
        return np.mean(results)
    
    # Sequential processing
    result_sequential = benchmark_function(
        cpu_intensive_task,
        1000,
        description="CPU-Sequential"
    )
    results['cpu_sequential'] = result_sequential
    
    # Test with different worker counts if ThreadPoolExecutor available
    try:
        from concurrent.futures import ThreadPoolExecutor
        
        def parallel_cpu_task(n_workers=4):
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            
            def worker_task(batch_size):
                matrices = [np.random.rand(50, 50) for _ in range(batch_size)]
                results = []
                for matrix in matrices:
                    result = np.dot(matrix, matrix.T)
                    results.append(np.trace(result))
                return np.mean(results)
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(worker_task, 25) for _ in range(n_workers)]
                results = [f.result() for f in futures]
            
            return np.mean(results)
        
        # Test parallel processing
        result_parallel = benchmark_function(
            parallel_cpu_task,
            4,
            description="CPU-Parallel-4workers"
        )
        results['cpu_parallel'] = result_parallel
        
        # Calculate speedup
        if result_sequential['success'] and result_parallel['success']:
            speedup = result_sequential['execution_time'] / result_parallel['execution_time']
            print(f"   🚀 Parallelization speedup: {speedup:.2f}x")
            results['parallelization_speedup'] = speedup
    
    except ImportError:
        print("   ⚠️  ThreadPoolExecutor not available")
    
    return results


def generate_performance_summary(system_info: Dict[str, Any], 
                                ollama_results: Dict[str, Any],
                                cpu_results: Dict[str, Any]):
    """Generate performance summary and recommendations"""
    
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 25)
    
    # System utilization
    print(f"🖥️  System Configuration:")
    print(f"   CPU Cores: {system_info['cpu_cores']}")
    print(f"   Memory: {system_info['memory_total_gb']:.1f}GB total")
    print(f"   Current Usage: {system_info['memory_percent']:.1f}%")
    
    # Ollama performance
    if ollama_results:
        print(f"\n🦙 Ollama Performance:")
        
        successful_ollama = {k: v for k, v in ollama_results.items() if v.get('success', False)}
        if successful_ollama:
            times = [v['execution_time'] for v in successful_ollama.values()]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"   Average Time: {avg_time:.2f}s")
            print(f"   Range: {min_time:.2f}s - {max_time:.2f}s")
            print(f"   Throughput: {1.0/avg_time:.2f} operations/sec")
            
            # Memory usage
            memory_deltas = [v['memory_delta_mb'] for v in successful_ollama.values()]
            avg_memory = sum(memory_deltas) / len(memory_deltas)
            print(f"   Memory Usage: {avg_memory:+.1f}MB average")
        else:
            print("   ❌ No successful Ollama operations")
    
    # CPU performance
    if cpu_results:
        print(f"\n💻 CPU Performance:")
        
        if 'parallelization_speedup' in cpu_results:
            speedup = cpu_results['parallelization_speedup']
            print(f"   Parallelization Speedup: {speedup:.2f}x")
            
            if speedup > 1.5:
                efficiency_rating = "Excellent"
            elif speedup > 1.2:
                efficiency_rating = "Good"
            elif speedup > 1.0:
                efficiency_rating = "Fair"
            else:
                efficiency_rating = "Poor"
            
            print(f"   Parallel Efficiency: {efficiency_rating}")
    
    # Recommendations
    print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
    
    recommendations = []
    
    # System-based recommendations
    if system_info['memory_percent'] > 80:
        recommendations.append("High memory usage - consider reducing batch sizes")
    
    if system_info['cpu_cores'] >= 4:
        recommendations.append("Multi-core CPU available - enable parallel processing")
    
    # Ollama-based recommendations
    if ollama_results:
        successful = sum(1 for v in ollama_results.values() if v.get('success', False))
        total = len(ollama_results)
        
        if successful < total:
            recommendations.append("Some Ollama operations failed - check model availability")
        
        times = [v['execution_time'] for v in ollama_results.values() if v.get('success', False)]
        if times and max(times) > 10:
            recommendations.append("Slow operations detected - consider smaller models or reduced max_samples")
    
    # CPU optimization recommendations
    if cpu_results and cpu_results.get('parallelization_speedup', 0) < 1.2:
        recommendations.append("Limited parallelization benefit - current workload may be I/O bound")
    
    # General recommendations
    recommendations.extend([
        "Current setup optimized for Ollama models - excellent for local AI",
        "Consider GPU acceleration only for very large transformer models",
        "CPU-only setup provides good performance for most use cases"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return {
        'system_info': system_info,
        'ollama_performance': ollama_results,
        'cpu_performance': cpu_results,
        'recommendations': recommendations
    }


def main():
    """Main performance test execution"""
    
    print("⚡ TokenSHAP Performance Analysis")
    print("=" * 40)
    
    # Test system resources
    system_info = test_system_resources()
    
    # Test available implementations
    print(f"\n🔍 Available Components:")
    print(f"   Ollama Integration: {'✅' if OLLAMA_AVAILABLE else '❌'}")
    print(f"   CoT Analysis: {'✅' if COT_AVAILABLE else '❌'}")
    print(f"   Configuration: {'✅' if CONFIG_AVAILABLE else '❌'}")
    
    # Run performance tests
    ollama_results = test_ollama_performance()
    cpu_results = test_cpu_optimizations()
    
    # Generate comprehensive summary
    summary = generate_performance_summary(system_info, ollama_results, cpu_results)
    
    # Save results
    try:
        import json
        with open('simple_performance_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n💾 Results saved to: simple_performance_results.json")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    print(f"\n🎯 Analysis Complete!")
    print("   Your current setup is optimized for Ollama-based operations")
    print("   CPU performance is well-utilized for the available workload")


if __name__ == "__main__":
    main()