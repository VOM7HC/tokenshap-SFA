"""
Example usage of the refactored TokenSHAP with SFA
"""

# Uncomment these imports when you have the dependencies installed:
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

from config import TokenSHAPConfig, AttributionMethod
from tokenshap_with_sfa import TokenSHAPWithSFA


def main():
    """Example usage of the TokenSHAP framework"""
    
    print("TokenSHAP with SFA - Example Usage")
    print("=" * 50)
    
    # Configuration
    config = TokenSHAPConfig(
        max_samples=50,  # Reduced for faster example
        batch_size=5,
        convergence_threshold=0.01,
        use_stratification=True,
        ensure_first_order=True,
        adaptive_convergence=True,
        parallel_workers=2,
        sfa_n_estimators=50,
        cot_max_steps=5
    )
    
    print("✓ Configuration created")
    
    # Note: Uncomment the following when you have the dependencies:
    """
    # Load model and tokenizer
    model_name = "gpt2"  # or any other model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Loaded model: {model_name}")
    
    # Initialize the explainer
    explainer = TokenSHAPWithSFA(model, tokenizer, config)
    print("✓ TokenSHAP explainer initialized")
    
    # Example prompts
    training_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The weather is sunny today with clear skies.",
        "Natural language processing enables computers to understand text."
    ]
    
    test_prompts = [
        "Deep learning models require large amounts of data.",
        "The cat sat on the mat in the afternoon sun."
    ]
    
    # Train SFA (this will take some time)
    print("\\nTraining SFA meta-learner...")
    training_result = explainer.train_sfa(training_prompts, batch_size=2)
    print(f"✓ SFA training completed: {training_result}")
    
    # Test different attribution methods
    test_prompt = test_prompts[0]
    print(f"\\nAnalyzing prompt: '{test_prompt}'")
    
    # TokenSHAP attribution
    print("\\n1. TokenSHAP attribution:")
    tokenshap_result = explainer.explain(
        test_prompt, 
        method=AttributionMethod.TOKENSHAP
    )
    for token, value in list(tokenshap_result.items())[:5]:  # Show first 5
        print(f"   {token}: {value:.4f}")
    
    # SFA attribution (faster)
    print("\\n2. SFA attribution:")
    sfa_result = explainer.explain(
        test_prompt,
        method=AttributionMethod.SFA
    )
    for token, value in list(sfa_result.items())[:5]:  # Show first 5
        print(f"   {token}: {value:.4f}")
    
    # Hybrid attribution
    print("\\n3. Hybrid attribution:")
    hybrid_result = explainer.explain(
        test_prompt,
        method=AttributionMethod.HYBRID
    )
    for token, value in list(hybrid_result.items())[:5]:  # Show first 5
        print(f"   {token}: {value:.4f}")
    
    # Chain-of-Thought attribution
    print("\\n4. CoT hierarchical attribution:")
    cot_result = explainer.explain(
        test_prompt,
        method=AttributionMethod.HYBRID,
        use_cot=True
    )
    
    if 'error' not in cot_result:
        print(f"   Reasoning depth: {cot_result['metrics']['reasoning_depth']}")
        print(f"   Chain coherence: {cot_result['metrics']['chain_coherence']:.4f}")
        print(f"   Critical steps: {len(cot_result['critical_steps'])}")
    
    # Benchmark performance
    print("\\n5. Performance benchmark:")
    benchmark_results = explainer.benchmark(test_prompts[:1])  # Test on 1 prompt
    for method, stats in benchmark_results.items():
        if method != 'speedup':
            print(f"   {method}: {stats['avg_time']:.4f}s avg")
    
    if 'speedup' in benchmark_results:
        print(f"   SFA speedup: {benchmark_results['speedup']:.2f}x")
    
    # Save the trained model
    explainer.save("tokenshap_sfa_model.pkl")
    print("\\n✓ Model saved to tokenshap_sfa_model.pkl")
    
    print("\\n✓ Example completed successfully!")
    """
    
    print("\nTo run this example:")
    print("1. Install dependencies: pip install torch transformers scikit-learn scipy")
    print("2. Uncomment the code section in this file")
    print("3. Run: python example_usage.py")


if __name__ == "__main__":
    main()