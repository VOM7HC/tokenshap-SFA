"""
CoT reasoning test with your actual available model: phi4-reasoning:latest
"""

import sys
sys.path.append('.')

from cot_ollama_reasoning import OllamaCoTAnalyzer, quick_cot_analysis
from config import TokenSHAPConfig


def main():
    print("🧠 Chain-of-Thought Analysis with phi4-reasoning")
    print("=" * 55)
    
    # Use your actual available model
    model_name = "phi4-reasoning:latest"
    api_url = "http://127.0.0.1:11434"
    
    print(f"📋 Using model: {model_name}")
    print(f"🔗 Server: {api_url}")
    
    # Test prompts that work well with reasoning models
    test_cases = [
        {
            "prompt": "If I have 12 apples and give away 3 to my friend, then buy 5 more, how many apples do I have?",
            "type": "Math Problem",
            "pattern": "mathematical"
        },
        {
            "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "type": "Logic Problem", 
            "pattern": "analytical"
        },
        {
            "prompt": "Should I save money or invest it? Consider the pros and cons.",
            "type": "Decision Analysis",
            "pattern": "comparative"
        }
    ]
    
    print(f"\n🚀 Quick CoT Analysis Tests")
    print("=" * 35)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['type']}")
        print(f"Prompt: '{test_case['prompt']}'")
        print("-" * 50)
        
        try:
            # Quick analysis
            result = quick_cot_analysis(
                test_case['prompt'], 
                model_name=model_name,
                api_url=api_url
            )
            print(result)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*55)
    
    # Detailed analysis test
    print(f"\n🔬 Detailed CoT Analysis Test")
    print("=" * 35)
    
    try:
        # Use fast config for testing
        config = TokenSHAPConfig(max_samples=3, batch_size=1)
        
        analyzer = OllamaCoTAnalyzer(
            model_name=model_name,
            api_url=api_url,
            config=config
        )
        
        detailed_prompt = "If a train travels at 60 mph for 2.5 hours, how far does it go?"
        print(f"📝 Detailed analysis prompt: '{detailed_prompt}'")
        print("🔄 Running analysis...")
        
        # Full analysis with token attribution
        detailed_result = analyzer.analyze_cot_attribution(
            detailed_prompt,
            pattern="phi4_style",  # Use phi4-specific pattern
            analyze_steps=True     # Include token-level analysis
        )
        
        if 'error' not in detailed_result:
            print("✅ Detailed analysis successful!")
            
            # Show visualization
            viz = analyzer.visualize_reasoning_analysis(detailed_result)
            print("\n📊 Visualization:")
            print(viz)
            
            # Show some key metrics
            metrics = detailed_result.get('metrics', {})
            print(f"\n📈 Key Metrics:")
            print(f"   Reasoning Steps: {metrics.get('reasoning_depth', 0)}")
            print(f"   Quality Score: {metrics.get('reasoning_quality', 0):.2f}/1.0")
            print(f"   Chain Coherence: {metrics.get('chain_coherence', 0):.2f}")
            print(f"   Tokens Analyzed: {metrics.get('total_tokens_analyzed', 0)}")
            
            # Show first few reasoning steps
            steps = detailed_result.get('reasoning_steps', [])
            if steps:
                print(f"\n🧠 First 2 Reasoning Steps:")
                for i, step in enumerate(steps[:2], 1):
                    print(f"   {i}. {step}")
        
        else:
            error_msg = detailed_result.get('error', 'Unknown error')
            print(f"❌ Detailed analysis failed: {error_msg}")
    
    except Exception as e:
        print(f"❌ Detailed analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*60)
    print("🎯 phi4-reasoning CoT Analysis Summary:")
    print("✅ Optimized for reasoning models like phi4")
    print("🧠 Extracts step-by-step thinking process")
    print("📊 Provides quality and coherence metrics")
    print("🔤 Token-level attribution within reasoning steps")
    print("🎨 Text-based visualization of reasoning chain")
    print("\n💡 Key advantages with phi4-reasoning:")
    print("   - Better structured reasoning steps")
    print("   - Higher quality thinking chains")
    print("   - More logical step progression")
    print("   - Explicit reasoning markers")
    print("="*60)


if __name__ == "__main__":
    main()