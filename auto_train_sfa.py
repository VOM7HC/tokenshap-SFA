"""
Auto-train SFA with enhanced training data and save for future use
Based on Claude Opus 4.1 suggestions with Ollama integration
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any

sys.path.append('.')

from config import TokenSHAPConfig
from tokenshap_ollama import TokenSHAPWithOllama
from ollama_integration import test_ollama_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_comprehensive_training_prompts() -> List[str]:
    """Get comprehensive training prompts for diverse SFA learning"""
    
    return [
        # Technical/Programming
        "Machine learning transforms data into insights",
        "Natural language processing enables AI understanding", 
        "Deep learning models learn complex patterns",
        "Computer vision recognizes images and objects",
        "Reinforcement learning optimizes decision making",
        "Python is a versatile programming language",
        "Neural networks process information like brains",
        "Algorithms solve computational problems efficiently",
        "Data structures organize information effectively",
        "Software engineering builds robust applications",
        
        # Mathematical/Analytical
        "Linear algebra underlies machine learning mathematics",
        "Statistics provide insights into data distributions",
        "Optimization algorithms find optimal solutions",
        "Probability theory models uncertainty quantification",
        "Calculus enables gradient-based learning methods",
        
        # Reasoning/Logic
        "If you study hard, then you will succeed",
        "Either it rains today or the sun shines",
        "All cats are animals, Fluffy is a cat",
        "The hypothesis explains the experimental results",
        "Evidence supports the scientific conclusion",
        
        # Everyday Language
        "The quick brown fox jumps over lazy dogs",
        "Reading books expands knowledge and imagination",
        "Exercise improves both physical and mental health",
        "Music brings joy and emotional expression",
        "Travel broadens perspectives and cultural understanding",
        
        # Business/Economics
        "Investment strategies diversify financial risk exposure",
        "Market demand influences product pricing decisions",
        "Innovation drives competitive business advantages",
        "Customer satisfaction increases brand loyalty significantly",
        "Automation improves operational efficiency and productivity",
        
        # Scientific/Research
        "Experimental methodology ensures reliable scientific results",
        "Peer review maintains research quality standards",
        "Hypothesis testing validates theoretical predictions",
        "Data analysis reveals hidden patterns and trends",
        "Reproducibility strengthens scientific credibility",
        
        # CoT-style reasoning prompts
        "What is 15 × 8? Let me calculate step by step",
        "How does photosynthesis work? First, plants capture sunlight",
        "Why is the sky blue? Light scattering explains this phenomenon", 
        "What causes earthquakes? Tectonic plate movements create seismic activity",
        "How do computers process information? Binary operations form the foundation",
    ]


def check_ollama_readiness(api_url: str, model_name: str) -> bool:
    """Check if Ollama is ready and wait for warmup"""
    print(" Checking Ollama server status...")
    
    if not test_ollama_connection(api_url):
        print(f" Error: Cannot connect to Ollama server at {api_url}")
        print(" Please start Ollama server: ollama serve")
        return False
    
    print(f" Ollama server is running at {api_url}")
    print(f" Model: {model_name}")
    
    # Wait for Ollama warmup
    print(" Waiting for Ollama model warmup...")
    print(" Note: First model request may take extra time for loading")
    warmup_time = 15  # seconds
    
    for i in range(warmup_time):
        print(f" Warmup progress: {i+1}/{warmup_time} seconds", end='\r')
        time.sleep(1)
    print()
    print(" Ollama warmup completed")
    
    return True

def auto_train_and_save(model_name: str = "phi4-reasoning:latest",
                       api_url: str = "http://127.0.0.1:11434",
                       batch_size: int = 5,
                       max_samples: int = 10) -> Dict[str, Any]:
    """
    Automatically train SFA with comprehensive data and save for future use
    
    Args:
        model_name: Ollama model to use
        api_url: Ollama API URL  
        batch_size: Training batch size
        max_samples: TokenSHAP samples per prompt (reduced for training speed)
    """
    
    print("Auto-Training Enhanced SFA with Ollama Integration")
    print("=" * 60)
    
    # Check Ollama readiness first
    if not check_ollama_readiness(api_url, model_name):
        return {'error': 'Ollama server not ready'}
    
    # Configure for training efficiency 
    config = TokenSHAPConfig(
        max_samples=max_samples,        # Reduced for faster training
        parallel_workers=2,             # Moderate parallelism for stability
        sfa_n_estimators=100,          # Rich SFA model
        sfa_max_depth=10,              # Deep learning capability
        cache_responses=True           # Cache for efficiency
    )
    
    print(f" Configuration:")
    print(f"   • Model: {model_name}")
    print(f"   • Max samples per prompt: {max_samples}")
    print(f"   • SFA estimators: {config.sfa_n_estimators}")
    print(f"   • Parallel workers: {config.parallel_workers}")
    
    # Initialize TokenSHAP with Ollama
    try:
        print(f"\n Initializing TokenSHAP with Ollama...")
        explainer = TokenSHAPWithOllama(
            model_name=model_name,
            api_url=api_url,
            config=config
        )
        print(f" TokenSHAP initialized successfully")
        
    except Exception as e:
        print(f" Failed to initialize TokenSHAP: {e}")
        return {'error': f'Initialization failed: {e}'}
    
    # Get comprehensive training prompts
    training_prompts = get_comprehensive_training_prompts()
    print(f"\n Training with {len(training_prompts)} diverse prompts:")
    
    # Show sample prompts
    for i, prompt in enumerate(training_prompts[:5], 1):
        print(f"   {i}. \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
    print(f"   ... and {len(training_prompts) - 5} more prompts")
    
    # Train SFA incrementally in batches for memory efficiency
    print(f"\n Starting SFA training in batches of {batch_size}...")
    start_time = time.time()
    
    total_batches = (len(training_prompts) + batch_size - 1) // batch_size
    all_results = []
    
    for batch_idx in range(0, len(training_prompts), batch_size):
        batch_num = (batch_idx // batch_size) + 1
        batch_prompts = training_prompts[batch_idx:batch_idx + batch_size]
        
        print(f"\n Training batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)...")
        print(" Processing with Ollama (this may take some time)...")
        batch_start = time.time()
        
        try:
            # Train SFA with this batch using 3-model approach
            result = explainer.train_sfa(batch_prompts)
            all_results.append(result)
            
            batch_time = time.time() - batch_start
            print(f"    Batch {batch_num} completed in {batch_time:.2f}s")
            
            if isinstance(result, dict):
                # Check for 3-model results
                if 'p_score' in result and 'shap_score' in result and 'p_shap_score' in result:
                    print(f"    3-Model Training Results:")
                    print(f"      P-only model: {result['p_score']:.4f}")
                    print(f"      SHAP-only model: {result['shap_score']:.4f}")
                    print(f"      P+SHAP model: {result['p_shap_score']:.4f}")
                    best_score = max(result['p_score'], result['shap_score'], result['p_shap_score'])
                    print(f"      Best ensemble score: {best_score:.4f}")
                # Fallback to standard results
                elif 'base_model_score' in result:
                    print(f"    Base model score: {result['base_model_score']:.4f}")
                    if 'augmented_model_score' in result:
                        print(f"    Augmented score: {result['augmented_model_score']:.4f}")
                        improvement = result['augmented_model_score'] - result['base_model_score']
                        print(f"    SFA improvement: {improvement:.4f}")
                    
        except Exception as e:
            print(f"    Batch {batch_num} failed: {e}")
            all_results.append({'error': str(e)})
            continue
        
        # Add waiting time between batches to prevent overloading
        if batch_num < total_batches:
            wait_time = 3
            print(f" Waiting {wait_time} seconds before next batch...")
            time.sleep(wait_time)
    
    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time:.2f} seconds")
    print(f" Average time per prompt: {total_time/len(training_prompts):.2f}s")
    
    # Get final SFA statistics using 3-model approach
    try:
        sfa_stats = explainer.sfa_learner.get_training_stats()
        print(f"\n Final SFA Statistics:")
        print(f"   • SFA trained: {sfa_stats.get('is_trained', False)}")
        print(f"   • Training samples: {sfa_stats.get('training_samples', 0)}")
        if sfa_stats.get('model_type') == '3_model_sfa':
            print(f"   • Model type: 3-Model SFA (P/SHAP/P+SHAP)")
            print(f"   • P-only score: {sfa_stats.get('p_score', 0):.4f}")
            print(f"   • SHAP-only score: {sfa_stats.get('shap_score', 0):.4f}")
            print(f"   • P+SHAP score: {sfa_stats.get('p_shap_score', 0):.4f}")
        else:
            print(f"   • Model type: {sfa_stats.get('model_type', 'Unknown')}")
            
    except Exception as e:
        print(f" Could not get SFA stats: {e}")
        sfa_stats = {}
    
    # Test augmented vs standard explanation
    print(f"\n Testing augmented vs standard explanation...")
    test_prompts = [
        "AI will revolutionize technology",
        "Data science drives business decisions", 
        "What is 12 × 7? Show step by step."
    ]
    
    for test_prompt in test_prompts:
        print(f"\n Test: \"{test_prompt}\"")
        
        try:
            # Test SFA explanation  
            sfa_result = explainer.explain(test_prompt, method="sfa", max_samples=3)
            print(f"    SFA result: {len(sfa_result)} tokens analyzed")
            
            # Show top attributed tokens
            if sfa_result:
                sorted_tokens = sorted(sfa_result.items(), key=lambda x: abs(x[1]), reverse=True)
                top_tokens = sorted_tokens[:3]
                for token, score in top_tokens:
                    print(f"      • '{token}': {score:.4f}")
                    
        except Exception as e:
            print(f"    Test failed: {e}")
    
    # Save final summary
    summary = {
        'training_prompts': len(training_prompts),
        'training_batches': total_batches,
        'total_time_seconds': total_time,
        'avg_time_per_prompt': total_time / len(training_prompts),
        'sfa_stats': sfa_stats,
        'successful_batches': sum(1 for r in all_results if 'error' not in r),
        'failed_batches': sum(1 for r in all_results if 'error' in r)
    }
    
    # Save the trained SFA model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    sfa_model_path = os.path.join(models_dir, "sfa_trained.pkl")
    
    try:
        explainer.save(sfa_model_path)
        print(f"\n SFA Model saved successfully to {sfa_model_path}")
    except Exception as e:
        print(f"\n Warning: Could not save SFA model: {e}")
    
    print(f"\n Training Summary:")
    print(f"   • Prompts processed: {summary['training_prompts']}")
    print(f"   • Successful batches: {summary['successful_batches']}/{total_batches}")
    print(f"   • Total training time: {summary['total_time_seconds']:.2f}s")
    print(f"   • SFA model saved: {sfa_model_path}")
    
    if summary['successful_batches'] > 0:
        print(f"\n Enhanced 3-Model SFA training completed successfully!")
        print(f" Your system now uses trained SFA data with P, SHAP, and P+SHAP ensemble!")
        print(f" Benefits: Improved accuracy through multi-model ensemble prediction")
    else:
        print(f"\n SFA training had issues - check the logs above")
        
    return summary


if __name__ == "__main__":
    # Full training mode
    auto_train_and_save()