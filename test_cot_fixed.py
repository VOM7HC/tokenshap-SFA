"""
Fixed CoT reasoning test - debugging the 404 issue
"""

import sys
sys.path.append('.')

import requests
from cot_ollama_reasoning import OllamaCoTAnalyzer
from config import TokenSHAPConfig


def test_ollama_connection():
    """Test basic Ollama connectivity and available models"""
    print("🔍 Testing Ollama Connection...")
    
    try:
        # Check server status
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama server is running")
            print(f"📋 Available models: {len(models)}")
            for model in models:
                print(f"   - {model['name']}")
            return models
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return []


def test_simple_generation():
    """Test simple text generation"""
    print("\n🧪 Testing Simple Generation...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "gemma3:270m",
                "prompt": "Hello! Please count to 3.",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            print(f"✅ Generation successful!")
            print(f"📝 Response: '{generated_text[:100]}...'")
            return True
        else:
            print(f"❌ Generation failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        return False


def test_cot_with_simple_prompt():
    """Test CoT with a very simple prompt"""
    print("\n🧠 Testing CoT with Simple Prompt...")
    
    try:
        config = TokenSHAPConfig(max_samples=2)  # Very lightweight
        analyzer = OllamaCoTAnalyzer(
            model_name="gemma3:270m",
            api_url="http://127.0.0.1:11434",
            config=config
        )
        
        # Use a simple prompt that should generate reasoning
        simple_prompt = "What is 2 + 3?"
        
        print(f"📝 Prompt: '{simple_prompt}'")
        
        # Try to generate reasoning
        steps, full_response, metadata = analyzer.generate_reasoning(
            simple_prompt, 
            pattern='mathematical',
            max_length=100  # Keep it short
        )
        
        print(f"📊 Results:")
        print(f"   Full response length: {len(full_response)}")
        print(f"   Number of steps: {len(steps)}")
        print(f"   Metadata: {metadata}")
        
        if full_response:
            print(f"📄 Full response: '{full_response}'")
        
        if steps:
            print(f"🔍 Reasoning steps:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")
        else:
            print("⚠️  No reasoning steps extracted")
        
        return len(steps) > 0
        
    except Exception as e:
        print(f"❌ CoT test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🔧 CoT Analysis Debugging")
    print("=" * 40)
    
    # Step 1: Test basic connectivity
    available_models = test_ollama_connection()
    
    if not available_models:
        print("\n❌ Cannot proceed - Ollama server not accessible")
        return
    
    # Step 2: Test simple generation
    generation_works = test_simple_generation()
    
    if not generation_works:
        print("\n❌ Cannot proceed - Basic generation not working")
        return
    
    # Step 3: Test CoT analysis
    cot_works = test_cot_with_simple_prompt()
    
    if cot_works:
        print(f"\n✅ CoT Analysis is working!")
        print(f"💡 You can now use it with more complex prompts")
    else:
        print(f"\n⚠️  CoT Analysis needs debugging")
        print(f"   The issue might be with reasoning step extraction")
    
    print(f"\n" + "=" * 50)
    print("🎯 Summary:")
    print(f"   Ollama Server: {'✅' if available_models else '❌'}")
    print(f"   Basic Generation: {'✅' if generation_works else '❌'}")
    print(f"   CoT Analysis: {'✅' if cot_works else '⚠️'}")


if __name__ == "__main__":
    main()