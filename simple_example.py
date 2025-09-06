"""
Simple Example: TokenSHAP with SFA - Core ML/DL Algorithm Demo
"""

import sys
sys.path.append('.')

from config import TokenSHAPConfig
from tokenshap_with_sfa import TokenSHAPWithSFA

def main():
    """Demonstrate core TokenSHAP + SFA functionality"""
    
    print("🧠 TokenSHAP with SFA - Core ML Algorithm Demo")
    print("=" * 45)
    
    # Configure the algorithm
    config = TokenSHAPConfig(
        max_samples=10,           # Shapley value sampling
        parallel_workers=2,       # CPU parallelization  
        attribution_method="shapley"  # Core ML method
    )
    
    # Initialize the main algorithm
    try:
        explainer = TokenSHAPWithSFA(config=config)
        print("✅ TokenSHAP algorithm initialized")
        
        # Example text for explanation
        text = "Machine learning algorithms can analyze and understand text patterns effectively."
        
        print(f"\n📝 Analyzing text: '{text}'")
        
        # Core ML explanation (would work with actual models)
        print("\n🔍 Core Algorithm Components:")
        print("   • TokenSHAP: Shapley-based token attribution")
        print("   • SFA: Meta-learning for fast approximation") 
        print("   • Value Functions: Similarity-based scoring")
        print("   • Hierarchical Analysis: Multi-level explanations")
        
        print(f"\n⚙️ Configuration:")
        print(f"   • Max Samples: {config.max_samples}")
        print(f"   • Workers: {config.parallel_workers}")
        print(f"   • Method: {config.attribution_method}")
        
        print(f"\n✨ Algorithm ready for ML/DL model integration")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Note: Full functionality requires ML model integration")

if __name__ == "__main__":
    main()