"""
Quick CoT test with phi4-reasoning - optimized for speed
"""

import sys
sys.path.append('.')

import requests
from cot_ollama_reasoning import OllamaCoTAnalyzer
from config import TokenSHAPConfig


def test_phi4_reasoning():
    print("⚡ Quick phi4-reasoning CoT Test")
    print("=" * 40)
    
    # Test basic generation first
    print("🧪 Testing basic generation...")
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "phi4-reasoning:latest",
                "prompt": "What is 2 + 3? Think step by step:",
                "stream": False,
                "options": {"num_predict": 50}  # Limit response length
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            print(f"✅ Basic generation works!")
            print(f"📝 Response preview: '{generated_text[:100]}...'")
        else:
            print(f"❌ Generation failed: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Generation test failed: {str(e)}")
        return
    
    # Test CoT analysis
    print(f"\n🧠 Testing CoT reasoning analysis...")
    
    try:
        # Very fast config
        config = TokenSHAPConfig(max_samples=1)  # Minimal for speed
        
        analyzer = OllamaCoTAnalyzer(
            model_name="phi4-reasoning:latest",
            api_url="http://127.0.0.1:11434",
            config=config
        )
        
        # Simple math problem
        prompt = "If I buy 3 apples for $2 each, how much do I spend?"
        print(f"📝 Test prompt: '{prompt}'")
        
        # Generate reasoning (without token analysis for speed)
        steps, full_response, metadata = analyzer.generate_reasoning(
            prompt, 
            pattern='phi4_style',
            max_length=200  # Limit response length
        )
        
        print(f"✅ CoT generation complete!")
        print(f"📊 Results:")
        print(f"   Response length: {len(full_response)} characters")
        print(f"   Reasoning steps extracted: {len(steps)}")
        print(f"   Quality score: {metadata.get('reasoning_quality', 0):.2f}")
        
        if full_response:
            print(f"\n📄 Generated response:")
            print(f"   '{full_response}'")
        
        if steps:
            print(f"\n🔍 Extracted reasoning steps:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")
        
        print(f"\n✅ phi4-reasoning CoT analysis is working!")
        
    except Exception as e:
        print(f"❌ CoT analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_phi4_reasoning()