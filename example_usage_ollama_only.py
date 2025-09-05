"""
Example usage of TokenSHAP with Ollama - No transformers required
"""

import sys
sys.path.append('.')

from config import TokenSHAPConfig
from tokenshap_ollama import TokenSHAPWithOllama


def main():
    """Example usage with your available Ollama model"""
    
    print("TokenSHAP with Ollama - Complete Example")
    print("=" * 50)
    
    # Use your available model
    model_name = "gemma3:270m"
    api_url = "http://127.0.0.1:11434"
    
    print(f"✓ Using model: {model_name}")
    print(f"✓ Server: {api_url}")
    
    # Configuration for practical use
    config = TokenSHAPConfig(
        max_samples=20,     # Good balance of speed vs accuracy
        batch_size=5,
        convergence_threshold=0.01,
        parallel_workers=1,  # Keep simple for Ollama
        cache_size=100      # Cache responses for efficiency
    )
    
    print("✓ Configuration created")
    
    # Initialize the explainer
    explainer = TokenSHAPWithOllama(
        model_name=model_name,
        api_url=api_url,
        config=config
    )
    
    print("✓ TokenSHAP explainer initialized")
    
    # Example prompts for different scenarios
    example_prompts = {
        "Simple": "The cat sits on the mat.",
        "Technical": "Machine learning algorithms process large datasets efficiently.",
        "Question": "What are the benefits of renewable energy sources?",
        "Creative": "The mysterious forest whispered ancient secrets to travelers."
    }
    
    print(f"\n🔍 Testing TokenSHAP explanations...")
    print("=" * 60)
    
    # Test explanations on different types of text
    all_results = {}
    
    for category, prompt in example_prompts.items():
        print(f"\n📝 {category} Text: '{prompt}'")
        
        try:
            result = explainer.explain(prompt)
            all_results[category] = result
            
            print(f"   🎯 Token Importance Rankings:")
            sorted_tokens = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for i, (token, importance) in enumerate(sorted_tokens, 1):
                symbol = "🔥" if importance > 0.5 else "⚡" if importance > 0.1 else "💫"
                print(f"      {i:2d}. {symbol} '{token:<12}' → {importance:+.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Training prompts for SFA
    training_prompts = [
        "Artificial intelligence transforms industries rapidly and efficiently.",
        "Deep learning models require substantial computational resources for training.",
        "Natural language processing enables computers to understand human communication.",
        "Data science combines statistics, programming, and domain expertise effectively.",
        "Cloud computing provides scalable infrastructure for modern applications."
    ]
    
    print(f"\n🎓 Training SFA Meta-Learner...")
    print("=" * 60)
    
    try:
        training_result = explainer.train_sfa(training_prompts)
        print(f"✅ SFA Training Complete!")
        print(f"   📊 Samples processed: {training_result.get('n_samples', 0)}")
        print(f"   📈 Cross-validation score: {training_result.get('cv_score', 'N/A')}")
        
        # Test SFA vs TokenSHAP speed
        print(f"\n⚡ Testing SFA Speed vs TokenSHAP...")
        
        test_prompt = "Python programming language offers excellent data analysis capabilities."
        print(f"   Test prompt: '{test_prompt}'")
        
        # Time TokenSHAP
        import time
        start = time.time()
        tokenshap_result = explainer.explain(test_prompt, method="tokenshap")
        tokenshap_time = time.time() - start
        
        # Time SFA
        start = time.time()
        sfa_result = explainer.explain(test_prompt, method="sfa")
        sfa_time = time.time() - start
        
        speedup = tokenshap_time / sfa_time if sfa_time > 0 else float('inf')
        
        print(f"\n   📊 Performance Comparison:")
        print(f"      TokenSHAP: {tokenshap_time:.2f}s")
        print(f"      SFA:       {sfa_time:.2f}s")
        print(f"      Speedup:   {speedup:.1f}x faster with SFA! 🚀")
        
        print(f"\n   🔍 Results Comparison:")
        print(f"      TokenSHAP top tokens:")
        ts_sorted = sorted(tokenshap_result.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (token, imp) in enumerate(ts_sorted[:3], 1):
            print(f"         {i}. '{token:<10}' → {imp:+.4f}")
            
        print(f"      SFA top tokens:")
        sfa_sorted = sorted(sfa_result.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (token, imp) in enumerate(sfa_sorted[:3], 1):
            print(f"         {i}. '{token:<10}' → {imp:+.4f}")
    
    except Exception as e:
        print(f"   ⚠️  SFA training had issues: {str(e)}")
        print(f"   ℹ️  This is normal with very small datasets")
    
    # Save the trained model
    print(f"\n💾 Saving trained model...")
    try:
        model_file = "tokenshap_ollama_trained.pkl"
        explainer.save(model_file)
        print(f"   ✅ Model saved to: {model_file}")
        
        # Test loading
        explainer.load(model_file)
        print(f"   ✅ Model loaded successfully")
        
    except Exception as e:
        print(f"   ⚠️  Save/load had issues: {str(e)}")
    
    # Final demonstration
    print(f"\n🎯 Final Demonstration...")
    print("=" * 60)
    
    demo_text = "Climate change requires immediate global action and cooperation."
    print(f"Demo text: '{demo_text}'")
    
    result = explainer.explain(demo_text)
    
    # Create a simple visualization
    print(f"\n📊 Token Importance Visualization:")
    max_importance = max(abs(v) for v in result.values())
    
    for token, importance in result.items():
        # Create a simple bar visualization
        bar_length = int(abs(importance) / max_importance * 20) if max_importance > 0 else 0
        bar = "█" * bar_length + "░" * (20 - bar_length)
        direction = "+" if importance >= 0 else "-"
        print(f"   '{token:<12}' |{bar}| {direction}{abs(importance):.4f}")
    
    print(f"\n" + "=" * 60)
    print("🌟 TokenSHAP + Ollama Integration Complete!")
    print("🚀 Key Features Demonstrated:")
    print("   ✓ Token-level importance analysis")
    print("   ✓ SFA training for speed improvement") 
    print("   ✓ Model saving and loading")
    print("   ✓ Multiple text types analysis")
    print("   ✓ Performance comparison")
    print("   ✓ Simple visualization")
    print(f"\n💡 Next Steps:")
    print("   - Try with longer, more complex texts")
    print("   - Experiment with different max_samples settings")
    print("   - Train SFA on domain-specific prompts")
    print("   - Pull more Ollama models for comparison")
    print("=" * 60)


if __name__ == "__main__":
    main()