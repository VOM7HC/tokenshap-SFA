"""
Working test with actual available Ollama model
"""

import sys
sys.path.append('.')

from config import TokenSHAPConfig
from tokenshap_ollama import TokenSHAPWithOllama

def main():
    print("TokenSHAP + Ollama Working Test")
    print("=" * 40)
    
    # Use the model that's actually available
    model_name = "gemma3:270m"  # This is what you have installed
    api_url = "http://127.0.0.1:11434"
    
    print(f"\n✓ Using available model: {model_name}")
    
    try:
        # Create a minimal configuration for faster testing
        config = TokenSHAPConfig(
            max_samples=5,      # Very small for quick test
            batch_size=1,
            parallel_workers=1
        )
        
        print("✓ Creating TokenSHAP explainer...")
        explainer = TokenSHAPWithOllama(
            model_name=model_name,
            api_url=api_url,
            config=config
        )
        
        print("✓ Explainer created successfully!")
        
        # Test with a simple prompt
        test_prompt = "Hello world"
        print(f"\n📝 Analyzing: '{test_prompt}'")
        
        result = explainer.explain(test_prompt)
        
        print(f"\n📊 Token Analysis Results:")
        print(f"   Found {len(result)} tokens")
        
        # Sort by absolute importance
        sorted_tokens = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (token, importance) in enumerate(sorted_tokens, 1):
            print(f"   {i:2d}. '{token:<8}' → {importance:+.6f}")
        
        print(f"\n🚀 Success! TokenSHAP is working with Ollama!")
        
        # Test SFA training if the basic test works
        print(f"\n🔬 Testing SFA training...")
        training_prompts = [
            "The sky is blue",
            "Water is wet", 
            "Fire is hot"
        ]
        
        print("   Training SFA meta-learner...")
        training_result = explainer.train_sfa(training_prompts)
        print(f"   ✓ SFA trained on {training_result.get('n_samples', 0)} samples")
        
        # Test SFA prediction
        print(f"\n⚡ Testing SFA fast prediction...")
        sfa_result = explainer.explain(test_prompt, method="sfa")
        
        print(f"   SFA Results:")
        sfa_sorted = sorted(sfa_result.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (token, importance) in enumerate(sfa_sorted, 1):
            print(f"   {i:2d}. '{token:<8}' → {importance:+.6f}")
        
        print(f"\n🎉 Complete success! Both TokenSHAP and SFA are working!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Check if model is available: ollama list") 
        print("   3. Pull the model if needed: ollama pull gemma3:270m")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n" + "="*50)
        print("🌟 Your TokenSHAP + Ollama integration is working perfectly!")
        print("🚀 You can now use it for real token-level explanations!")
        print("="*50)
    else:
        print(f"\n" + "="*50) 
        print("⚠️  Setup needs attention - check the tips above")
        print("="*50)