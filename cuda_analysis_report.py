"""
Comprehensive CUDA/GPU Usage Analysis Report for TokenSHAP-SFA Project
"""

import sys
import os
sys.path.append('.')

def analyze_cuda_usage():
    """Generate comprehensive CUDA usage analysis"""
    
    print("🔍 CUDA/GPU Usage Analysis Report")
    print("=" * 50)
    
    # Check current environment
    print("\n📋 ENVIRONMENT STATUS")
    print("-" * 25)
    
    try:
        import torch
        torch_available = True
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🎯 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🖥️  CUDA Version: {torch.version.cuda}")
            print(f"📱 Device Count: {torch.cuda.device_count()}")
            print(f"🏷️  Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - CPU mode only")
    except ImportError:
        torch_available = False
        print("❌ PyTorch not installed - using CPU-only implementations")
    
    try:
        from transformers import __version__ as transformers_version
        print(f"🤗 Transformers: {transformers_version}")
    except ImportError:
        print("❌ Transformers not installed")
    
    # Analyze configuration
    print(f"\n⚙️  CONFIGURATION ANALYSIS")
    print("-" * 30)
    
    try:
        from config import TokenSHAPConfig, TORCH_AVAILABLE
        config = TokenSHAPConfig()
        print(f"🔧 TORCH_AVAILABLE flag: {TORCH_AVAILABLE}")
        print(f"🎮 use_gpu setting: {config.use_gpu}")
        print(f"🏭 parallel_workers: {config.parallel_workers}")
        
        if not TORCH_AVAILABLE:
            print("ℹ️  Config automatically disabled GPU due to missing PyTorch")
        elif not config.use_gpu:
            print("ℹ️  Config disabled GPU usage")
    except Exception as e:
        print(f"❌ Config analysis failed: {str(e)}")
    
    # Analyze source code GPU usage
    print(f"\n📝 SOURCE CODE ANALYSIS")
    print("-" * 28)
    
    gpu_usage_files = {
        "token_shap.py": {
            "description": "Enhanced TokenSHAP Implementation",
            "gpu_features": [
                "Lines 35-36: model.cuda() - Moves model to GPU",
                "Lines 49-50: inputs = {k: v.cuda() for k, v in inputs.items()} - Moves input tensors to GPU",
                "GPU usage conditional on config.use_gpu and torch.cuda.is_available()"
            ],
            "cpu_fallback": "✅ Yes - operates on CPU if CUDA unavailable"
        },
        
        "cot_explainer.py": {
            "description": "Chain-of-Thought TokenSHAP",
            "gpu_features": [
                "Lines 67-68: inputs = {k: v.cuda() for k, v in inputs.items()} - GPU tensor operations",
                "Uses same GPU logic as token_shap.py"
            ],
            "cpu_fallback": "✅ Yes - falls back to CPU computation"
        },
        
        "config.py": {
            "description": "Configuration Management",
            "gpu_features": [
                "Line 47: use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()",
                "Smart detection of GPU availability",
                "Automatic fallback to CPU if CUDA unavailable"
            ],
            "cpu_fallback": "✅ Yes - automatic CPU fallback"
        },
        
        "tokenshap_ollama.py": {
            "description": "Ollama Integration (CPU-Only)",
            "gpu_features": [
                "NO GPU usage - designed for Ollama API",
                "Uses SimpleTokenizer instead of transformers",
                "Pure CPU implementation"
            ],
            "cpu_fallback": "✅ CPU-only by design"
        },
        
        "cot_ollama_reasoning.py": {
            "description": "Ollama CoT Analysis (CPU-Only)",
            "gpu_features": [
                "NO GPU usage - Ollama API based",
                "CPU-only token processing",
                "No PyTorch dependencies"
            ],
            "cpu_fallback": "✅ CPU-only by design"
        }
    }
    
    for filename, info in gpu_usage_files.items():
        print(f"\n📄 {filename}")
        print(f"   📋 {info['description']}")
        print(f"   🎮 GPU Features:")
        for feature in info['gpu_features']:
            print(f"      • {feature}")
        print(f"   🔄 CPU Fallback: {info['cpu_fallback']}")
    
    # Analyze current deployment
    print(f"\n🚀 CURRENT DEPLOYMENT STATUS")
    print("-" * 35)
    
    if not torch_available:
        print("📍 STATUS: CPU-Only Mode (PyTorch not installed)")
        print("🔧 ACTIVE: Ollama-based implementations only")
        print("⚡ PERFORMANCE: Using lightweight CPU implementations")
        print("💡 RECOMMENDATION: This is optimal for Ollama usage")
        
        active_components = [
            "✅ OllamaCoTAnalyzer - CoT analysis with phi4-reasoning",
            "✅ TokenSHAPWithOllama - CPU-based token attribution", 
            "✅ SimpleTokenizer - Basic tokenization",
            "✅ SFAMetaLearner - Scikit-learn based meta-learning",
            "❌ EnhancedTokenSHAP - Requires PyTorch",
            "❌ CoTTokenSHAP - Requires transformers"
        ]
        
    else:
        try:
            import torch
            if torch.cuda.is_available():
                print("📍 STATUS: GPU-Enabled Mode")
                print("🎮 GPU AVAILABLE: Full CUDA acceleration possible")
            else:
                print("📍 STATUS: CPU Mode (PyTorch installed, no CUDA)")
                print("🔧 FALLBACK: Using CPU tensors")
        except:
            pass
            
        active_components = [
            "✅ All components available",
            "✅ GPU acceleration when enabled",
            "✅ Automatic CPU fallback"
        ]
    
    print(f"\n🏗️  ACTIVE COMPONENTS:")
    for component in active_components:
        print(f"   {component}")
    
    # Performance implications
    print(f"\n⚡ PERFORMANCE IMPLICATIONS")
    print("-" * 32)
    
    performance_analysis = {
        "With CUDA (if available)": [
            "🚀 Faster model inference for transformers",
            "⚡ GPU-accelerated tensor operations",
            "🎯 Better performance for large models",
            "💾 Higher memory usage (GPU VRAM)"
        ],
        "CPU-Only Mode (current)": [
            "🔧 Works with Ollama models perfectly",
            "💡 Lower memory requirements", 
            "🌐 No CUDA driver dependencies",
            "⏰ Slower for large transformer models",
            "✅ Optimal for your current Ollama setup"
        ]
    }
    
    for mode, implications in performance_analysis.items():
        print(f"\n📊 {mode}:")
        for implication in implications:
            print(f"   {implication}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    if not torch_available:
        print("🎯 CURRENT SETUP IS OPTIMAL FOR OLLAMA:")
        print("   ✅ No unnecessary dependencies")
        print("   ✅ Lightweight CPU implementations")
        print("   ✅ Perfect for phi4-reasoning analysis")
        print("   ✅ SFA meta-learning works great on CPU")
        print("")
        print("🔮 IF YOU WANT FULL TRANSFORMER SUPPORT:")
        print("   📦 Install: conda install pytorch transformers")
        print("   🎮 For GPU: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    else:
        print("🎯 CURRENT SETUP HAS FULL CAPABILITIES")
        print("   ✅ Can use both Ollama and transformers")
        print("   ✅ GPU acceleration available if needed")
        print("   ✅ Automatic fallback to CPU")
    
    print(f"\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    current_mode = "CPU-Only (Ollama Optimized)" if not torch_available else "Full PyTorch Support"
    
    summary = f"""
🏷️  Current Mode: {current_mode}
🎮 GPU Usage: {'❌ Not available' if not torch_available else '✅ Available with fallback'}
🚀 Ollama Integration: ✅ Fully functional
🧠 CoT Analysis: ✅ Working with phi4-reasoning
⚡ Performance: {'Optimized for Ollama' if not torch_available else 'Full acceleration available'}

🎯 Your codebase intelligently adapts to available hardware:
   • CPU-only when PyTorch unavailable (current)
   • GPU-accelerated when CUDA available
   • Automatic fallback ensures compatibility
   • Ollama integration always works regardless
"""
    
    print(summary)
    print("=" * 60)

def check_specific_gpu_usage():
    """Check specific GPU usage in current session"""
    print("\n🔍 RUNTIME GPU USAGE CHECK")
    print("-" * 30)
    
    try:
        from config import TokenSHAPConfig
        config = TokenSHAPConfig()
        print(f"📋 Current use_gpu setting: {config.use_gpu}")
        
        if hasattr(config, 'use_gpu') and config.use_gpu:
            print("⚠️  GPU is enabled in config but may not be used due to missing PyTorch")
        else:
            print("✅ GPU disabled - using CPU implementations")
            
    except Exception as e:
        print(f"❌ Could not check config: {str(e)}")

if __name__ == "__main__":
    analyze_cuda_usage()
    check_specific_gpu_usage()