"""
Example usage of TokenSHAP with Ollama models
"""

from config import TokenSHAPConfig
from tokenshap_ollama import TokenSHAPWithOllama
from ollama_integration import test_ollama_connection
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example usage of TokenSHAP with Ollama models"""
    
    print("TokenSHAP with Ollama - Example Usage")
    print("=" * 50)
    
    # Using phi4-reasoning model (GPU accelerated)
    models_to_test = [
        {
            "name": "phi4-reasoning:latest",
            "url": "http://127.0.0.1:11434",
            "description": "Local phi4-reasoning model (RTX 4090 optimized)"
        }
    ]
    
    # Test connection to servers
    print("\n1. Testing Ollama server connections...")
    for model_config in models_to_test:
        is_available = test_ollama_connection(model_config["url"])
        status = "✓ Available" if is_available else "✗ Not available"
        print(f"   {model_config['description']}: {status}")
    
    # Configuration optimized for phi4-reasoning (large model)
    config = TokenSHAPConfig(
        max_samples=5,  # Reduced significantly for phi4-reasoning performance
        batch_size=2,   # Smaller batches for memory efficiency
        convergence_threshold=0.1,  # Less strict for faster results
        parallel_workers=1,  # Single worker for stability with large model
        cache_responses=True,  # Cache to avoid re-computation
    )
    
    # Example prompts
    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Python is a great programming language.",
    ]
    
    training_prompts = [
        "Natural language processing enables AI understanding.",
        "Deep learning models learn complex patterns.",
        "Computer vision recognizes images and objects.",
    ]
    
    # Test phi4-reasoning model with proper timeout handling
    for model_config in models_to_test:
        print(f"\n2. Testing {model_config['description']}...")
        print("=" * 60)
        print("💡 phi4-reasoning is a large model (14.7B parameters)")
        print("⏰ Expected time: 1-3 minutes with GPU acceleration")
        print("🔥 First analysis may take extra time for warmup...")
        
        try:
            # Initialize TokenSHAP with phi4-reasoning
            print("\n🔄 Initializing TokenSHAP with phi4-reasoning...")
            explainer = TokenSHAPWithOllama(
                model_name=model_config["name"],
                api_url=model_config["url"],
                config=config
            )
            
            print(f"✓ Initialized TokenSHAP with {model_config['name']}")
            
            # Test basic explanation with simple prompt
            test_prompt = test_prompts[0]
            print(f"\n⚡ Analyzing: '{test_prompt}'")
            print("🔄 Processing with phi4-reasoning (please be patient)...")
            
            result = explainer.explain(test_prompt, max_samples=3)  # Reduced for phi4-reasoning
            
            print("   Token importance rankings:")
            sorted_tokens = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (token, importance) in enumerate(sorted_tokens[:7]):
                print(f"     {i+1:2d}. '{token}': {importance:+.4f}")
            
            # Train SFA for faster predictions
            print(f"\n   Training SFA meta-learner...")
            training_result = explainer.train_sfa(training_prompts)
            print(f"   ✓ SFA training completed: {training_result.get('n_samples', 0)} samples")
            
            # Test SFA prediction
            print(f"\n   Testing SFA prediction...")
            sfa_result = explainer.explain(test_prompt, method="sfa")
            
            print("   SFA Token importance rankings:")
            sfa_sorted = sorted(sfa_result.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (token, importance) in enumerate(sfa_sorted[:5]):
                print(f"     {i+1:2d}. '{token}': {importance:+.4f}")
            
            # Benchmark performance
            print(f"\n   Benchmarking performance...")
            benchmark_results = explainer.benchmark(test_prompts[:2])  # Test 2 prompts
            
            for method, stats in benchmark_results.items():
                if method == 'speedup':
                    print(f"     SFA Speedup: {stats:.2f}x faster")
                else:
                    print(f"     {method.capitalize()}: {stats['avg_time']:.2f}s average")
            
            # Save trained model
            model_filename = f"tokenshap_{model_config['name'].replace(':', '_').replace('.', '_')}.pkl"
            explainer.save(model_filename)
            print(f"   ✓ Model saved to {model_filename}")
            
            print(f"\n✓ {model_config['description']} testing completed successfully!")
            
        except Exception as e:
            print(f"✗ Error testing {model_config['description']}: {str(e)}")
            logger.error(f"Model test failed: {str(e)}", exc_info=True)
    
    print(f"\n" + "=" * 60)
    print("Example Summary:")
    print("- TokenSHAP can work with any Ollama model")
    print("- SFA provides significant speedup after training")
    print("- Models can be saved and loaded for reuse")
    print("- Both local and remote Ollama servers are supported")
    print("\nNext steps:")
    print("1. Install Ollama: https://ollama.ai/")
    print("2. Pull models: ollama pull phi4-reasoning:latest")
    print("3. Run this example with your models")
    

def simple_usage_example():
    """Simple usage example for phi4-reasoning with proper timeouts"""
    print("\nSimple Usage Example with phi4-reasoning:")
    print("-" * 45)
    
    # Setup for phi4-reasoning model (requires longer timeouts)
    model_name = "phi4-reasoning:latest"  # GPU-optimized reasoning model
    api_url = "http://127.0.0.1:11434"  # your Ollama server
    
    print(f"💡 Using {model_name} (14.7B parameters)")
    print("⏰ Expected time: 30-60 seconds with GPU acceleration")
    print("🔥 First run may take extra time for model warmup...")
    
    try:
        # Initialize with reduced samples for phi4-reasoning
        print("\n🔄 Initializing TokenSHAP with phi4-reasoning...")
        explainer = TokenSHAPWithOllama(
            model_name=model_name,
            api_url=api_url,
            config=TokenSHAPConfig(
                max_samples=3,  # Reduced for phi4-reasoning performance
                parallel_workers=1,  # Single worker for stability
                convergence_threshold=0.1  # Less strict for faster results
            )
        )
        
        # Explain a simple prompt
        prompt = "AI will change the world."
        print(f"\n⚡ Analyzing prompt: '{prompt}'")
        print("🔄 Processing with phi4-reasoning (please be patient)...")
        
        result = explainer.explain(prompt)
        
        print(f"\n✅ Analysis completed!")
        print("📊 Token importance scores:")
        for token, importance in result.items():
            print(f"  '{token}': {importance:+.3f}")
        
        print("\n🎯 phi4-reasoning analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        if "timeout" in str(e).lower():
            print("💡 phi4-reasoning timed out - this is normal for large models")
            print("💡 Try increasing timeout or use a smaller model for testing")
        else:
            print("💡 Make sure Ollama is running: ollama serve")
            print("💡 Verify model is available: ollama list")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed with error: {str(e)}")
        print("This might be due to:")
        print("- Ollama server not running")
        print("- Model not available") 
        print("- Network connectivity issues")
        
        # Run simple example as fallback
        simple_usage_example()