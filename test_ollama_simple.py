"""
Simple test of Ollama integration
"""

import sys
sys.path.append('.')

from config import TokenSHAPConfig
from tokenshap_ollama import TokenSHAPWithOllama
from ollama_integration import test_ollama_connection

def main():
    print("Simple TokenSHAP + Ollama Test")
    print("=" * 40)
    
    # Test configurations from your setup
    test_configs = [
        {"name": "gemma2:2b", "url": "http://127.0.0.1:11434", "description": "Local Gemma"},
        {"name": "llama3.2-vision:latest", "url": "http://35.95.163.15:11434", "description": "Remote Llama Vision"},
    ]
    
    # Test server connections
    print("\n1. Testing server connections...")
    for config in test_configs:
        try:
            available = test_ollama_connection(config["url"])
            status = "✓ Available" if available else "✗ Not available"
            print(f"   {config['description']}: {status}")
        except Exception as e:
            print(f"   {config['description']}: ✗ Error - {str(e)}")
    
    # Test with the first available model
    print("\n2. Testing TokenSHAP integration...")
    
    for config in test_configs:
        try:
            print(f"\n   Testing {config['description']}...")
            
            # Create a very lightweight configuration for testing
            test_config = TokenSHAPConfig(max_samples=3, batch_size=1)  # Minimal for testing
            
            explainer = TokenSHAPWithOllama(
                model_name=config["name"],
                api_url=config["url"],
                config=test_config
            )
            
            print(f"   ✓ Created explainer for {config['name']}")
            
            # Test a very simple prompt
            test_prompt = "AI is useful."
            print(f"   Testing prompt: '{test_prompt}'")
            
            result = explainer.explain(test_prompt)
            
            print(f"   ✓ Got {len(result)} token explanations:")
            for token, importance in result.items():
                print(f"     '{token}': {importance:.4f}")
            
            print(f"   ✓ {config['description']} test successful!")
            break  # If one works, that's enough for the test
            
        except Exception as e:
            print(f"   ✗ {config['description']} failed: {str(e)}")
            continue
    
    print("\n✓ Simple test completed!")

if __name__ == "__main__":
    main()