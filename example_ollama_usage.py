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
    
    # Your specified configurations
    models_to_test = [
        {
            "name": "gemma2:2b",
            "url": "http://127.0.0.1:11434",
            "description": "Local Gemma 2B model"
        },
        {
            "name": "llama3.2-vision:latest", 
            "url": "http://35.95.163.15:11434",
            "description": "Remote Llama 3.2 Vision model"
        }
    ]
    
    # Test connection to servers
    print("\n1. Testing Ollama server connections...")
    for model_config in models_to_test:
        is_available = test_ollama_connection(model_config["url"])
        status = "✓ Available" if is_available else "✗ Not available"
        print(f"   {model_config['description']}: {status}")
    
    # Configuration for faster testing
    config = TokenSHAPConfig(
        max_samples=20,  # Reduced for faster demo
        batch_size=5,
        convergence_threshold=0.05,
        parallel_workers=1  # Keep it simple for Ollama
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
    
    # Test each available model
    for model_config in models_to_test:
        print(f"\n2. Testing {model_config['description']}...")
        print("=" * 60)
        
        try:
            # Initialize TokenSHAP with Ollama
            explainer = TokenSHAPWithOllama(
                model_name=model_config["name"],
                api_url=model_config["url"],
                config=config
            )
            
            print(f"✓ Initialized TokenSHAP with {model_config['name']}")
            
            # Test basic explanation
            test_prompt = test_prompts[0]
            print(f"\n   Analyzing: '{test_prompt}'")
            
            result = explainer.explain(test_prompt, max_samples=10)
            
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
    print("2. Pull models: ollama pull gemma2:2b")
    print("3. Run this example with your models")
    

def simple_usage_example():
    """Simple usage example for quick testing"""
    print("\nSimple Usage Example:")
    print("-" * 30)
    
    # Quick setup - adjust model and URL as needed
    model_name = "gemma2:2b"  # or "llama3.2:3b", etc.
    api_url = "http://127.0.0.1:11434"  # your Ollama server
    
    try:
        # Initialize (will use simple tokenizer if transformers not available)
        explainer = TokenSHAPWithOllama(
            model_name=model_name,
            api_url=api_url,
            config=TokenSHAPConfig(max_samples=5)  # Very fast for testing
        )
        
        # Explain a simple prompt
        prompt = "AI will change the world."
        result = explainer.explain(prompt)
        
        print(f"Prompt: '{prompt}'")
        print("Token importance:")
        for token, importance in result.items():
            print(f"  '{token}': {importance:+.3f}")
        
        print("\n✓ Simple example completed!")
        
    except Exception as e:
        print(f"✗ Simple example failed: {str(e)}")
        print("Make sure Ollama is running and the model is available.")


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