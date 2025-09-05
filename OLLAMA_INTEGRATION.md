# Ollama Integration for TokenSHAP

This document explains how to use TokenSHAP with Ollama models, making it easy to run explainable AI on local or remote Ollama servers.

## New Files Added

### Core Integration Files
- **`ollama_integration.py`** - Ollama model classes and utilities
- **`tokenshap_ollama.py`** - TokenSHAP implementation optimized for Ollama
- **`example_ollama_usage.py`** - Complete usage examples

### Key Classes

#### `TokenSHAPWithOllama`
Main interface for using TokenSHAP with Ollama models.

```python
from tokenshap_ollama import TokenSHAPWithOllama

# Initialize with your model
explainer = TokenSHAPWithOllama(
    model_name="gemma2:2b",
    api_url="http://127.0.0.1:11434"
)

# Explain any text
result = explainer.explain("Your text here")
```

#### `OllamaModel` & `SimpleOllamaModel`
Direct Ollama API integration with support for text and vision models.

```python
from ollama_integration import OllamaModel

# For vision models
model = OllamaModel("llama3.2-vision:latest", "http://35.95.163.15:11434")
response = model.generate("Describe this image", image_path="image.jpg")

# For text models  
model = OllamaModel("gemma2:2b")
response = model.generate("What is AI?")
```

## Usage Examples

### Basic Usage (Your Configuration)

```python
# Local model
explainer = TokenSHAPWithOllama(
    model_name="gemma2:2b",
    api_url="http://127.0.0.1:11434"
)

# Remote model  
explainer = TokenSHAPWithOllama(
    model_name="llama3.2-vision:latest",
    api_url="http://35.95.163.15:11434"
)

# Analyze text
prompt = "Machine learning will transform industries."
result = explainer.explain(prompt)

# Show most important tokens
for token, importance in sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"'{token}': {importance:.4f}")
```

### Training SFA for Speed

```python
# Train on example prompts for faster future explanations
training_prompts = [
    "AI is changing the world rapidly.",
    "Deep learning requires large datasets.", 
    "Natural language models understand context."
]

training_result = explainer.train_sfa(training_prompts)

# Now use SFA for much faster explanations
fast_result = explainer.explain("New prompt here", method="sfa")
```

### Benchmarking Performance

```python
test_prompts = ["Test prompt 1", "Test prompt 2"]
benchmark = explainer.benchmark(test_prompts)

print(f"TokenSHAP: {benchmark['tokenshap']['avg_time']:.2f}s")
print(f"SFA: {benchmark['sfa']['avg_time']:.2f}s")  
print(f"Speedup: {benchmark['speedup']:.1f}x")
```

## Key Features

### 1. **No Transformers Dependency**
The Ollama integration works without PyTorch or Transformers:
```bash
pip install numpy scikit-learn requests
```

### 2. **Automatic Tokenization**
Uses simple but effective word-based tokenization when transformers isn't available.

### 3. **Local and Remote Support**
Works with both local Ollama servers and remote endpoints.

### 4. **Vision Model Support**
`OllamaModel` class supports vision models with image inputs.

### 5. **SFA Acceleration**
Train once, then get 10-100x speedup on explanations.

## Installation & Setup

### Option 1: Minimal Setup (Ollama only)
```bash
pip install numpy scikit-learn requests
```

### Option 2: Full Setup (with Transformers)
```bash
pip install numpy scikit-learn requests torch transformers
```

### Running Examples

1. **Start Ollama server** (if local):
   ```bash
   ollama serve
   ```

2. **Pull your models**:
   ```bash
   ollama pull gemma2:2b
   ollama pull llama3.2-vision:latest
   ```

3. **Run the example**:
   ```bash
   python example_ollama_usage.py
   ```

## Migration from Original Code

If you were using the classes mentioned in your code:

### Old Way:
```python
# Your original approach
ollama_model = OllamaModel(model_name="llama3.2-vision:latest", api_url="http://35.95.163.15:11434")
tfidf_embedding = TfidfTextVectorizer()
token_shap_ollama = TokenSHAP(model=ollama_model, splitter=splitter, vectorizer=tfidf_embedding)
```

### New Way:
```python
# Simplified with new integration
explainer = TokenSHAPWithOllama(
    model_name="llama3.2-vision:latest",
    api_url="http://35.95.163.15:11434"
)

# Everything is built-in now
result = explainer.explain("Your prompt here")
```

## Configuration Options

```python
from config import TokenSHAPConfig

# Customize behavior
config = TokenSHAPConfig(
    max_samples=50,      # Reduce for faster computation
    batch_size=5,        # Ollama works well with smaller batches  
    cache_size=1000,     # Cache responses for efficiency
    parallel_workers=1   # Keep simple for Ollama
)

explainer = TokenSHAPWithOllama(
    model_name="your-model",
    config=config
)
```

## Error Handling

The integration includes robust error handling:

```python
try:
    result = explainer.explain("Your prompt")
except Exception as e:
    print(f"Explanation failed: {str(e)}")
    # Common causes:
    # - Ollama server not running
    # - Model not available
    # - Network issues
```

## Performance Tips

1. **Use SFA after training** - 10-100x speedup
2. **Reduce max_samples** for faster initial computation
3. **Cache responses** - enabled by default
4. **Use local models** when possible for better latency

## Troubleshooting

### Model Not Found
```bash
ollama pull your-model-name
```

### Server Connection Issues
- Check if Ollama is running: `curl http://localhost:11434/api/tags`
- Verify the API URL and port
- Check firewall settings for remote servers

### Memory Issues
- Use smaller models (e.g., `gemma2:2b` instead of larger variants)
- Reduce `max_samples` in configuration
- Clear response cache periodically

This integration makes TokenSHAP much more accessible by removing heavy dependencies while maintaining full functionality!