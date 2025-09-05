"""
Quick example of TokenSHAP with Ollama - Fast version
"""

import sys
sys.path.append('.')

from config import TokenSHAPConfig
from tokenshap_ollama import TokenSHAPWithOllama


def main():
    print("Quick TokenSHAP + Ollama Demo")
    print("=" * 35)
    
    # Fast configuration
    config = TokenSHAPConfig(
        max_samples=5,    # Very fast for demo
        batch_size=1
    )
    
    explainer = TokenSHAPWithOllama(
        model_name="gemma3:270m",
        api_url="http://127.0.0.1:11434", 
        config=config
    )
    
    print("✓ Explainer ready")
    
    # Quick test
    prompt = "Python is powerful"
    print(f"\nAnalyzing: '{prompt}'")
    
    result = explainer.explain(prompt)
    
    print("Results:")
    for token, importance in result.items():
        print(f"  '{token}': {importance:.4f}")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()