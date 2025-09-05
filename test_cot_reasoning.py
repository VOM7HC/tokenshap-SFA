"""
Test Chain-of-Thought reasoning analysis with Ollama models
"""

import sys
sys.path.append('.')

from cot_ollama_reasoning import OllamaCoTAnalyzer, quick_cot_analysis
from config import TokenSHAPConfig


def main():
    print("🧠 Chain-of-Thought Reasoning Analysis Test")
    print("=" * 50)
    
    # Use your available model for testing (adjust if you have phi4-reasoning)
    available_model = "gemma3:270m"  # Your current model
    api_url = "http://127.0.0.1:11434"
    
    print(f"📋 Using model: {available_model}")
    print(f"🔗 Server: {api_url}")
    
    # Test prompts that benefit from step-by-step reasoning
    test_prompts = [
        {
            "prompt": "If I have 12 apples and give away 3 to my friend, then buy 5 more, how many apples do I have?",
            "type": "Math Problem",
            "pattern": "mathematical"
        },
        {
            "prompt": "Why is the sky blue?",
            "type": "Scientific Question", 
            "pattern": "analytical"
        },
        {
            "prompt": "Should I invest in renewable energy stocks? Consider the pros and cons.",
            "type": "Decision Making",
            "pattern": "comparative"
        }
    ]
    
    # Test 1: Quick analysis
    print(f"\n🚀 Quick CoT Analysis Test")
    print("-" * 30)
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {test_case['type']}")
        print(f"Prompt: '{test_case['prompt']}'")
        
        try:
            # Quick analysis (without detailed token analysis for speed)
            result = quick_cot_analysis(
                test_case['prompt'], 
                model_name=available_model,
                api_url=api_url
            )
            print(result)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*50)
    
    # Test 2: Detailed analysis
    print(f"\n🔬 Detailed CoT Analysis Test")
    print("-" * 30)
    
    try:
        # Create analyzer with fast configuration
        config = TokenSHAPConfig(max_samples=3)  # Very fast for demo
        analyzer = OllamaCoTAnalyzer(
            model_name=available_model,
            api_url=api_url,
            config=config
        )
        
        # Choose a simple problem for detailed analysis
        detailed_prompt = test_prompts[0]['prompt']  # Math problem
        
        print(f"📝 Analyzing: '{detailed_prompt}'")
        print("🔄 Running detailed analysis...")
        
        # Full analysis with token-level attribution
        detailed_result = analyzer.analyze_cot_attribution(
            detailed_prompt,
            pattern="mathematical",
            analyze_steps=True  # Include token analysis
        )
        
        if 'error' not in detailed_result:
            print("✅ Analysis complete!")
            
            # Show detailed results
            viz = analyzer.visualize_reasoning_analysis(detailed_result)
            print(viz)
            
            # Show some raw data
            print("\n📊 Raw Analysis Data:")
            print(f"   Steps generated: {len(detailed_result.get('reasoning_steps', []))}")
            print(f"   Token attributions: {len(detailed_result.get('token_attributions', []))}")
            print(f"   Critical steps: {len(detailed_result.get('critical_steps', []))}")
            
            # Show first reasoning step
            steps = detailed_result.get('reasoning_steps', [])
            if steps:
                print(f"\n🔍 First reasoning step:")
                print(f"   '{steps[0][:100]}...'")
        
        else:
            print(f"❌ Detailed analysis failed: {detailed_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Detailed analysis error: {str(e)}")
    
    print(f"\n" + "="*60)
    print("🎯 CoT Analysis Summary:")
    print("✅ Quick analysis: Text-based reasoning visualization")
    print("🔬 Detailed analysis: Token-level attribution + step importance")
    print("🧠 Reasoning patterns: Mathematical, analytical, comparative, etc.")
    print("📊 Quality metrics: Coherence, depth, complexity scores")
    print(f"\n💡 To use with phi4-reasoning model:")
    print("   1. Pull model: ollama pull phi4-reasoning:latest")
    print("   2. Change model_name in code to 'phi4-reasoning:latest'")
    print("   3. Phi4 will provide higher quality reasoning steps!")
    print("="*60)


if __name__ == "__main__":
    main()