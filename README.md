# TokenSHAP with SFA - Core ML/DL Algorithm

## 🧠 Streamlined Codebase

This repository now contains only the **core ML/DL algorithms** and essential functionality, with all unnecessary files removed.

## 📁 Core Files (12 total)

### **Core ML/DL Algorithm Components:**
- `token_shap.py` - Enhanced TokenSHAP algorithm (Shapley value computation)
- `sfa_learner.py` - SFA Meta-Learner (machine learning meta-learning)
- `value_functions.py` - ML value functions (similarity, probability)
- `cot_explainer.py` - Chain-of-Thought ML explainer
- `tokenshap_with_sfa.py` - Main unified ML framework

### **Configuration & Utilities:**
- `config.py` - Algorithm configuration and parameters
- `utils.py` - Core utilities (caching, token processing)
- `__init__.py` - Package interface

### **Examples & Integration:**
- `simple_example.py` - Basic ML algorithm demonstration  
- **`cot_ollama_reasoning.py`** - **Chain-of-Thought analysis with phi4-reasoning (GPU optimized)** ⭐
- `cot_analysis_example.py` - Extended CoT analysis examples
- `example_ollama_usage.py` - Ollama model integration example

### **Dependencies:**
- `requirements.txt` - Python package requirements

## 🗑️ Removed Files (18+ files)

**Performance Testing & Benchmarking:**
- `performance_benchmark.py`
- `simple_performance_test.py`
- `cpu_performance_test.py`
- `minimal_performance_test.py`
- `cuda_analysis_report.py`

**Redundant Optimization Files:**
- `optimized_performance_manager.py`
- `cot_explainer_optimized.py`
- `ollama_integration.py` (merged functionality)
- `tokenshap_ollama.py` (merged functionality)

**Test & Demo Files:**
- `test_*.py` (9 test files)
- `demo_*.py` 
- `quick_*.py`

**Generated Files:**
- All `.json` result files
- Performance analysis `.md` files

## 🚀 Core ML Algorithm Features

### **TokenSHAP Algorithm:**
- Shapley value-based token attribution
- Enhanced sampling strategies
- Parallel computation support

### **SFA Meta-Learning:**
- Fast approximation of Shapley values
- Scikit-learn based meta-learner
- Adaptive sampling strategies

### **Chain-of-Thought Analysis:**
- Hierarchical reasoning analysis
- Step-by-step attribution
- Multi-level explanations

### **Value Functions:**
- Similarity-based scoring
- Probability distributions
- Flexible value computation

## 💻 Usage

```python
from config import TokenSHAPConfig
from tokenshap_with_sfa import TokenSHAPWithSFA

# Configure ML algorithm
config = TokenSHAPConfig(
    max_samples=50,
    attribution_method="shapley"
)

# Initialize core algorithm
explainer = TokenSHAPWithSFA(config=config)
```

## 📊 What Was Kept vs Removed

| **Kept (Core ML/DL)** | **Removed (Non-Essential)** |
|------------------------|------------------------------|
| 11 files | 18+ files |
| ~50KB total | ~200KB+ removed |
| Core algorithms only | Tests, benchmarks, duplicates |
| Essential examples | Redundant examples |
| Clean architecture | Performance analysis files |

The codebase is now **focused, efficient, and maintainable** with only the essential ML/DL algorithm components.