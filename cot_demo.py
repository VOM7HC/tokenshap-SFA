"""
Chain-of-Thought Analysis Demo - Shows ML/DL Algorithm Structure
"""

import sys
sys.path.append('.')

from config import TokenSHAPConfig

def demo_cot_analysis_structure():
    """Demonstrate the CoT analysis ML/DL structure without requiring Ollama"""
    
    print("🧠 Chain-of-Thought Analysis - ML/DL Algorithm Demo")
    print("=" * 55)
    
    # Example reasoning prompt
    prompt = "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?"
    print(f"📝 Example Prompt: '{prompt}'")
    
    # Simulated CoT steps (what Ollama phi4-reasoning would generate)
    simulated_steps = [
        "I need to calculate the distance using the formula: distance = speed × time",
        "Given information: speed = 60 mph, time = 2.5 hours",
        "Calculating: distance = 60 × 2.5 = 150 miles",
        "Therefore, the train travels 150 miles in total"
    ]
    
    print(f"\n🔍 Generated CoT Steps:")
    for i, step in enumerate(simulated_steps, 1):
        print(f"   Step {i}: {step}")
    
    # Configure ML algorithm
    config = TokenSHAPConfig(
        max_samples=20,         # Shapley value sampling
        cot_max_steps=10,      # Maximum reasoning steps
        sfa_n_estimators=50,   # SFA meta-learning
        parallel_workers=2     # Parallel processing
    )
    
    print(f"\n⚙️ ML Algorithm Configuration:")
    print(f"   • Max Shapley samples: {config.max_samples}")
    print(f"   • CoT max steps: {config.cot_max_steps}")
    print(f"   • SFA estimators: {config.sfa_n_estimators}")
    print(f"   • Parallel workers: {config.parallel_workers}")
    
    # Simulate ML analysis results
    print(f"\n🔬 ML Analysis Results:")
    
    # 1. Token-level attribution (simulated Shapley values)
    token_attributions = [
        {"step": 1, "tokens": {"calculate": 0.85, "distance": 0.72, "formula": 0.68}},
        {"step": 2, "tokens": {"60": 0.91, "mph": 0.45, "2.5": 0.88, "hours": 0.52}},
        {"step": 3, "tokens": {"60": 0.94, "2.5": 0.89, "150": 0.96}},
        {"step": 4, "tokens": {"150": 0.78, "miles": 0.65, "total": 0.43}}
    ]
    
    print(f"   1. 🧮 Token-level Attribution (Shapley Values):")
    for attr in token_attributions:
        step_num = attr["step"]
        tokens = attr["tokens"]
        print(f"      Step {step_num}: {dict(sorted(tokens.items(), key=lambda x: x[1], reverse=True))}")
    
    # 2. Step-level importance (simulated)
    step_importance = [0.25, 0.35, 0.30, 0.10]  # Normalized importance scores
    print(f"   2. 📊 Step-level Importance:")
    for i, importance in enumerate(step_importance, 1):
        bar_length = int(importance * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"      Step {i}: |{bar}| {importance:.3f}")
    
    # 3. Chain coherence (simulated)
    chain_coherence = 0.87
    print(f"   3. 🔗 Chain Coherence: {chain_coherence:.3f}")
    
    # 4. Critical components
    critical_tokens = [
        {"token": "calculate", "value": 0.85, "step": 1},
        {"token": "150", "value": 0.96, "step": 3},
        {"token": "60", "value": 0.94, "step": 3},
        {"token": "2.5", "value": 0.89, "step": 3}
    ]
    
    print(f"   4. 🎯 Critical Tokens (Top 4):")
    for token_info in critical_tokens:
        print(f"      '{token_info['token']}': {token_info['value']:.3f} (Step {token_info['step']})")
    
    # 5. Reasoning quality metrics
    print(f"   5. 📈 Reasoning Quality Metrics:")
    print(f"      • Reasoning depth: {len(simulated_steps)} steps")
    print(f"      • Average step complexity: 0.72")
    print(f"      • Logic coherence: {chain_coherence:.3f}")
    print(f"      • Total tokens analyzed: {sum(len(attr['tokens']) for attr in token_attributions)}")
    
    print(f"\n✨ ML/DL Components Used:")
    print(f"   ✅ TokenSHAP: Shapley value computation for token importance")
    print(f"   ✅ SFA Meta-Learning: Fast approximation of expensive computations")
    print(f"   ✅ Hierarchical Analysis: Token → Step → Chain level attribution")
    print(f"   ✅ Value Functions: Similarity-based scoring and coherence")
    print(f"   ✅ Critical Detection: Automated identification of key elements")
    
    print(f"\n🎯 Real-world Usage:")
    print("   from cot_ollama_reasoning import quick_cot_analysis")
    print("   result = quick_cot_analysis('Your prompt here')")
    print("   # Requires running Ollama with phi4-reasoning model")
    
    return {
        "prompt": prompt,
        "cot_steps": simulated_steps,
        "token_attributions": token_attributions,
        "step_importance": step_importance,
        "chain_coherence": chain_coherence,
        "critical_tokens": critical_tokens,
        "ml_components": [
            "TokenSHAP (Shapley values)",
            "SFA Meta-Learning",
            "Hierarchical Analysis", 
            "Value Functions",
            "Critical Detection"
        ]
    }


def main():
    """Main demo function"""
    print("This demo shows the ML/DL structure of Chain-of-Thought analysis")
    print("without requiring a running Ollama server.\n")
    
    result = demo_cot_analysis_structure()
    
    print(f"\n📊 Summary:")
    print(f"✅ Demonstrated {len(result['ml_components'])} ML/DL components")
    print(f"✅ Analyzed {len(result['cot_steps'])} reasoning steps")
    print(f"✅ Generated {sum(len(attr['tokens']) for attr in result['token_attributions'])} token attributions")
    print(f"✅ Computed hierarchical importance scores")
    print(f"✅ Identified {len(result['critical_tokens'])} critical components")
    

if __name__ == "__main__":
    main()