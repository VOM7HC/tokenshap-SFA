"""
Chain-of-Thought Analysis Example - Core ML/DL CoT Algorithm Demo
"""

import sys
sys.path.append('.')

from cot_ollama_reasoning import OllamaCoTAnalyzer, quick_cot_analysis
from config import TokenSHAPConfig

def demo_quick_cot_analysis():
    """Demonstrate quick CoT analysis functionality"""
    
    print("🧠 Quick CoT Analysis Demo")
    print("=" * 30)
    
    # Example reasoning prompts
    test_prompts = [
        "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
        "What are the main benefits of renewable energy compared to fossil fuels?",
        "Explain step by step how machine learning algorithms learn from data."
    ]
    
    print("📝 Testing Chain-of-Thought reasoning analysis...")
    print("💡 Using phi4-reasoning (14.7B parameters) - GPU accelerated")
    print("⏰ Expected time per example: 15-30 seconds with RTX 4090")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔍 Example {i}:")
        print(f"Prompt: '{prompt[:60]}...' " if len(prompt) > 60 else f"Prompt: '{prompt}'")
        
        # Use quick analysis function with phi4-reasoning
        try:
            print("🔄 Processing with phi4-reasoning (please be patient)...")
            result = quick_cot_analysis(
                prompt, 
                model_name="phi4-reasoning:latest",
                api_url="http://127.0.0.1:11434"
            )
            print("✅ Analysis completed")
            print("📊 Result preview:", result[:200] + "..." if len(result) > 200 else result)
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)[:100]}...")
            if "timeout" in str(e).lower():
                print("💡 phi4-reasoning timed out - this is normal for large models")
                print("💡 Try a simpler prompt or check GPU memory")
            else:
                print("💡 Make sure Ollama is running: ollama serve")
                print("💡 Verify model is available: ollama list")


def demo_detailed_cot_analysis():
    """Demonstrate detailed CoT analysis with ML components"""
    
    print("\n🔬 Detailed CoT Analysis Demo")
    print("=" * 35)
    
    # Configure the ML algorithm
    config = TokenSHAPConfig(
        max_samples=20,           # Shapley sampling for token attribution
        parallel_workers=2,       # Parallel processing
        cot_max_steps=10,        # Maximum reasoning steps
        sfa_n_estimators=50      # SFA meta-learning estimators
    )
    
    print("⚙️ ML Algorithm Configuration:")
    print(f"   • Max Shapley samples: {config.max_samples}")
    print(f"   • CoT max steps: {config.cot_max_steps}")
    print(f"   • SFA estimators: {config.sfa_n_estimators}")
    print(f"   • Parallel workers: {config.parallel_workers}")
    
    try:
        # Initialize the CoT analyzer
        analyzer = OllamaCoTAnalyzer(
            model_name="phi4-reasoning:latest",
            config=config
        )
        
        print("\n✅ CoT Analyzer initialized with ML components:")
        print("   • TokenSHAP integration for token-level attribution")
        print("   • SFA meta-learning for fast approximation")
        print("   • Hierarchical analysis (token → step → chain levels)")
        print("   • Value functions for reasoning quality assessment")
        
        # Example analysis
        example_prompt = "What is the most efficient way to sort a list of numbers?"
        
        print(f"\n📝 Analyzing: '{example_prompt}'")
        print("\n🔍 ML Analysis Components:")
        print("   1. Chain-of-Thought Generation (reasoning steps)")
        print("   2. Token-level Attribution (Shapley values)")
        print("   3. Step-level Importance (hierarchical analysis)")
        print("   4. Chain Coherence (reasoning quality)")
        print("   5. Critical Component Identification")
        
        # Demonstrate the analysis structure
        result = analyzer.analyze_cot_attribution(
            example_prompt,
            analyze_steps=True,    # Full token analysis
            use_sfa=True          # Use fast SFA approximation
        )
        
        print("✅ Analysis completed successfully")
        print("📊 Generated ML Analysis Structure:")
        print(f"   • Reasoning steps: {len(result.get('reasoning_steps', []))}")
        print(f"   • Token attributions: {len(result.get('token_attributions', []))}")
        print(f"   • Step importance scores: Available")
        print(f"   • Chain coherence: {result.get('chain_coherence', 'N/A')}")
        
    except Exception as e:
        print(f"⚠️  Detailed analysis demo (requires Ollama): {e}")
        print("💡 This demonstrates the ML structure - full functionality needs Ollama server")


def demo_cot_ml_components():
    """Demonstrate the ML/DL components in CoT analysis"""
    
    print("\n🤖 CoT ML/DL Components Demo")
    print("=" * 32)
    
    print("📋 Core ML Algorithm Components in CoT Analysis:")
    
    print("\n1. 🧮 TokenSHAP Integration:")
    print("   • Shapley value computation for token importance")
    print("   • Game theory-based attribution")
    print("   • Parallel processing for efficiency")
    
    print("\n2. 🎯 SFA Meta-Learning:")
    print("   • Fast approximation of expensive Shapley computations")
    print("   • Scikit-learn based regression models")
    print("   • Adaptive sampling strategies")
    
    print("\n3. 🏗️ Hierarchical Analysis:")
    print("   • Token-level: Individual word/token importance")
    print("   • Step-level: Reasoning step significance") 
    print("   • Chain-level: Overall reasoning coherence")
    
    print("\n4. 📊 Value Functions:")
    print("   • Similarity-based scoring between reasoning steps")
    print("   • Quality assessment of reasoning chains")
    print("   • Coherence measurement algorithms")
    
    print("\n5. 🔍 Critical Component Detection:")
    print("   • Automated identification of key reasoning elements")
    print("   • Threshold-based importance filtering")
    print("   • Ranking and prioritization of components")
    
    print("\n✨ ML Benefits:")
    print("   • Quantitative analysis of reasoning quality")
    print("   • Explainable AI for chain-of-thought processes")
    print("   • Scalable analysis for large reasoning datasets")
    print("   • Integration with modern transformer models")


def main():
    """Main CoT analysis demonstration"""
    
    print("🧠 Chain-of-Thought Analysis - Core ML/DL Algorithm")
    print("=" * 55)
    
    print("This demonstrates the ML/DL components for Chain-of-Thought analysis:")
    print("• TokenSHAP algorithm for token attribution")
    print("• SFA meta-learning for efficient computation") 
    print("• Hierarchical reasoning analysis")
    print("• Value functions for quality assessment")
    
    # Run demonstrations
    demo_quick_cot_analysis()
    demo_detailed_cot_analysis() 
    demo_cot_ml_components()
    
    print("\n🎯 Summary:")
    print("✅ CoT analysis combines multiple ML/DL techniques")
    print("✅ Provides quantitative reasoning quality assessment")
    print("✅ Scales efficiently with SFA meta-learning")
    print("✅ Integrates with modern language models")
    
    print("\n💡 Usage:")
    print("   from cot_ollama_reasoning import quick_cot_analysis")
    print("   result = quick_cot_analysis('Your reasoning prompt here')")


if __name__ == "__main__":
    main()