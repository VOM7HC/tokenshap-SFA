"""
Demonstration of Chain-of-Thought analysis for phi4-reasoning
"""

import sys
sys.path.append('.')

from cot_ollama_reasoning import OllamaCoTAnalyzer, quick_cot_analysis
from config import TokenSHAPConfig


def demo_with_available_model():
    """Demo using your currently available model"""
    print("🧠 CoT Analysis Demo with Available Model")
    print("=" * 45)
    
    # Use your available model
    model_name = "gemma3:270m"
    
    test_problems = [
        "If a pizza has 8 slices and I eat 3, how many are left?",
        "Why do leaves change color in autumn?",
        "What are the main causes of climate change?"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n📝 Problem {i}: {problem}")
        print("-" * 40)
        
        try:
            result = quick_cot_analysis(problem, model_name=model_name)
            print(result)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()


def demo_phi4_reasoning():
    """Demo specifically for phi4-reasoning model"""
    print("🚀 CoT Analysis Demo for phi4-reasoning")
    print("=" * 45)
    
    # This requires phi4-reasoning model to be available
    model_name = "phi4-reasoning:latest"
    
    # More complex problems that benefit from reasoning models
    reasoning_problems = [
        {
            "problem": "A farmer has chickens and cows. In total, there are 30 heads and 74 legs. How many chickens and how many cows are there?",
            "type": "Logic Puzzle"
        },
        {
            "problem": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "type": "Logical Reasoning"
        },
        {
            "problem": "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons?",
            "type": "Problem Solving"
        }
    ]
    
    print("ℹ️  Note: This demo requires phi4-reasoning model")
    print("   To install: ollama pull phi4-reasoning:latest\n")
    
    for i, test_case in enumerate(reasoning_problems, 1):
        print(f"📝 {test_case['type']} {i}:")
        print(f"   {test_case['problem']}")
        print()
        
        try:
            # Create detailed analyzer
            config = TokenSHAPConfig(max_samples=5)
            analyzer = OllamaCoTAnalyzer(
                model_name=model_name,
                config=config
            )
            
            # Perform analysis
            result = analyzer.analyze_cot_attribution(
                test_case['problem'],
                pattern='phi4_style',
                analyze_steps=False  # Skip token analysis for speed
            )
            
            if 'error' not in result:
                # Show reasoning steps
                steps = result.get('reasoning_steps', [])
                print(f"🧠 Reasoning Steps ({len(steps)} total):")
                for j, step in enumerate(steps[:3], 1):  # Show first 3 steps
                    print(f"   {j}. {step[:80]}...")
                
                # Show metrics
                metrics = result.get('metrics', {})
                quality = metrics.get('reasoning_quality', 0)
                coherence = metrics.get('chain_coherence', 0)
                print(f"\n📊 Quality: {quality:.2f}, Coherence: {coherence:.2f}")
            else:
                print(f"❌ Analysis failed: {result.get('error')}")
                
        except Exception as e:
            print(f"⚠️  Could not connect to {model_name}: {str(e)}")
            print(f"   Make sure the model is installed and Ollama is running")
        
        print("-" * 50)


def main():
    """Main demonstration function"""
    
    print("🎯 TokenSHAP Chain-of-Thought Analysis")
    print("🔗 Optimized for Ollama Reasoning Models")
    print("=" * 50)
    
    # Demo 1: With available model
    demo_with_available_model()
    
    print("\n" + "=" * 60)
    
    # Demo 2: With phi4-reasoning (if available)
    demo_phi4_reasoning()
    
    print(f"\n" + "=" * 60)
    print("🌟 Key Features of CoT Analysis:")
    print("✅ Step-by-step reasoning extraction")
    print("✅ Step importance scoring") 
    print("✅ Token-level attribution within steps")
    print("✅ Reasoning quality assessment")
    print("✅ Chain coherence analysis")
    print("✅ Critical step identification")
    print("\n💡 Usage Examples:")
    print("```python")
    print("from cot_ollama_reasoning import OllamaCoTAnalyzer")
    print("")
    print("# Quick analysis")
    print("result = quick_cot_analysis('Your problem', 'phi4-reasoning:latest')")
    print("")
    print("# Detailed analysis")
    print("analyzer = OllamaCoTAnalyzer('phi4-reasoning:latest')")
    print("detailed = analyzer.analyze_cot_attribution('Your problem')")
    print("```")
    print("=" * 60)


if __name__ == "__main__":
    main()