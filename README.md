# TokenSHAP with SFA - Refactored Structure

This repository contains a refactored version of the TokenSHAP with SFA (Shapley-based Feature Augmentation) framework. The original monolithic file has been split into multiple focused modules for better organization, maintainability, and reusability.

## File Structure

### Core Modules

- **`config.py`** - Configuration classes and enums
  - `TokenSHAPConfig`: Main configuration dataclass
  - `AttributionMethod`: Enum for different attribution methods

- **`utils.py`** - Utility classes
  - `ThreadSafeCache`: Thread-safe LRU cache implementation
  - `TokenProcessor`: Handles tokenization and token processing

- **`value_functions.py`** - Value function implementations
  - `ValueFunction`: Abstract base class
  - `SimilarityValueFunction`: TF-IDF cosine similarity based
  - `ProbabilityValueFunction`: Model log probability based

- **`token_shap.py`** - Core TokenSHAP implementation
  - `EnhancedTokenSHAP`: Main TokenSHAP class with improved sampling and convergence

- **`sfa_learner.py`** - SFA Meta-Learner
  - `SFAMetaLearner`: Shapley-based Feature Augmentation meta-learner

- **`cot_explainer.py`** - Chain-of-Thought explainer
  - `CoTTokenSHAP`: Chain-of-Thought aware hierarchical TokenSHAP

- **`tokenshap_with_sfa.py`** - Main unified interface
  - `TokenSHAPWithSFA`: Unified framework combining all components

### Supporting Files

- **`__init__.py`** - Package initialization with convenient imports
- **`requirements.txt`** - Required dependencies
- **`example_usage.py`** - Example usage demonstration
- **`README.md`** - This documentation file

## Key Benefits of Refactoring

1. **Modularity**: Each component is now in its own file, making the code easier to understand and maintain
2. **Reusability**: Individual components can be imported and used independently
3. **Testing**: Each module can be tested separately
4. **Extensibility**: New value functions, attribution methods, or utilities can be added easily
5. **Readability**: Smaller, focused files are easier to navigate and understand

## Usage

### Basic Usage

```python
from tokenshap_with_sfa import TokenSHAPWithSFA
from config import TokenSHAPConfig, AttributionMethod
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure
config = TokenSHAPConfig(max_samples=100, batch_size=10)

# Initialize explainer
explainer = TokenSHAPWithSFA(model, tokenizer, config)

# Explain a prompt
result = explainer.explain("Your prompt here", method=AttributionMethod.TOKENSHAP)
```

### Advanced Usage with SFA Training

```python
# Train SFA for faster explanations
training_prompts = ["prompt1", "prompt2", "prompt3"]
explainer.train_sfa(training_prompts)

# Use SFA for faster explanations
result = explainer.explain("Your prompt here", method=AttributionMethod.SFA)
```

### Chain-of-Thought Explanations

```python
# Get hierarchical CoT attribution
cot_result = explainer.explain(
    "Your prompt here", 
    method=AttributionMethod.HYBRID,
    use_cot=True
)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Import and use:
```python
from tokenshap_with_sfa import TokenSHAPWithSFA
```

## Migration from Original File

If you were using the original monolithic `tokenshap_sfa.py` file:

1. Replace `from tokenshap_sfa import TokenSHAPWithSFA` with `from tokenshap_with_sfa import TokenSHAPWithSFA`
2. All other APIs remain the same
3. You can now import individual components if needed:
   ```python
   from token_shap import EnhancedTokenSHAP
   from sfa_learner import SFAMetaLearner
   from cot_explainer import CoTTokenSHAP
   ```

## Development

To extend the framework:

1. **Add new value functions**: Inherit from `ValueFunction` in `value_functions.py`
2. **Add new attribution methods**: Extend `AttributionMethod` enum in `config.py`
3. **Add utilities**: Add to `utils.py`
4. **Modify configuration**: Update `TokenSHAPConfig` in `config.py`

## Testing

Run syntax checks on all files:
```bash
python -m py_compile *.py
```

For full testing with dependencies:
```bash
python example_usage.py
```