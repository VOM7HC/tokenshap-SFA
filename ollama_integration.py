"""
Ollama model integration for TokenSHAP with SFA
"""

import requests
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """Abstract base class for language models"""
    
    def __init__(self, model_name: str, api_url: str = None):
        self.model_name = model_name
        self.api_url = api_url
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the model"""
        pass


class SimpleOllamaModel(ModelBase):
    """Simple Ollama client for basic text generation"""
    
    def __init__(self, model_name: str, api_url: str = "http://127.0.0.1:11434"):
        super().__init__(model_name, api_url)
    
    def generate(self, prompt: str, max_length: int = 256, temperature: float = 0.7, 
                 stream: bool = False, **kwargs) -> str:
        """Generate response from Ollama model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_predict": max_length,
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=600  # Extended timeout for phi4-reasoning (10 minutes)
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise Exception(f"Error connecting to Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise Exception(f"Error generating response: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the Ollama server and model are available"""
        try:
            # Check server health
            response = requests.get(f"{self.api_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            return self.model_name in model_names
            
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {str(e)}")
            return False


class OllamaModel(ModelBase):
    """Advanced Ollama model implementation supporting both text and vision"""
    
    def __init__(self, model_name: str, api_url: str = "http://127.0.0.1:11434"):
        super().__init__(model_name, api_url)
        self._simple_client = SimpleOllamaModel(model_name, api_url)
    
    def generate(self, prompt: str, image_path: Optional[str] = None, 
                 max_length: int = 256, temperature: float = 0.7, **kwargs) -> str:
        """Generate response with optional image input"""
        
        if image_path:
            # For vision models, we need to encode the image
            return self._generate_with_image(prompt, image_path, max_length, temperature)
        else:
            # Use simple text generation
            return self._simple_client.generate(prompt, max_length, temperature)
    
    def _generate_with_image(self, prompt: str, image_path: str, 
                           max_length: int, temperature: float) -> str:
        """Generate response with image input (for vision models)"""
        try:
            import base64
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "num_predict": max_length,
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=600  # Extended timeout for phi4-reasoning (10 minutes)
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            # Fall back to text-only generation
            return self._simple_client.generate(prompt, max_length, temperature)
        except Exception as e:
            logger.error(f"Vision generation error: {str(e)}")
            raise Exception(f"Error generating response with image: {str(e)}")
    
    def is_vision_model(self) -> bool:
        """Check if the model supports vision"""
        vision_keywords = ["vision", "llava", "minicpm", "cogvlm"]
        return any(keyword in self.model_name.lower() for keyword in vision_keywords)
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        return self._simple_client.is_available()


class OllamaModelAdapter:
    """Adapter to make Ollama models compatible with TokenSHAP framework"""
    
    def __init__(self, ollama_model: OllamaModel):
        self.ollama_model = ollama_model
        self.model_name = ollama_model.model_name
    
    def generate(self, input_ids=None, attention_mask=None, max_length=256, 
                 temperature=0.7, do_sample=True, **kwargs):
        """Generate method compatible with transformers-style interface"""
        # Extract prompt from input_ids if provided
        if hasattr(input_ids, 'shape') and input_ids is not None:
            # For tensor inputs, we need a tokenizer to decode
            # This is a simplified approach - in practice you'd need proper tokenization
            prompt = str(input_ids)
        else:
            # Fallback - this adapter assumes prompt is passed directly
            prompt = kwargs.get('prompt', 'Hello, world!')
        
        response = self.ollama_model.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        # Return in a format that mimics transformers output
        return MockGenerationOutput(response)
    
    def cuda(self):
        """Mock cuda method for compatibility"""
        return self
    
    def to(self, device):
        """Mock to method for compatibility"""
        return self


class MockGenerationOutput:
    """Mock output to mimic transformers generation output"""
    
    def __init__(self, text_response: str):
        self.text_response = text_response
        # Create mock tensor-like object
        self.sequences = [MockTensor(text_response)]
    
    def __getitem__(self, index):
        if index == 0:
            return MockTensor(self.text_response)
        raise IndexError("Mock output only supports index 0")


class MockTensor:
    """Mock tensor for compatibility with existing decode methods"""
    
    def __init__(self, text: str):
        self.text = text
    
    def __getitem__(self, index):
        return self.text


# Factory function for easy model creation
def create_ollama_model(model_name: str, api_url: str = "http://127.0.0.1:11434", 
                       simple: bool = False) -> ModelBase:
    """Factory function to create Ollama models"""
    if simple:
        return SimpleOllamaModel(model_name, api_url)
    else:
        return OllamaModel(model_name, api_url)


# Pre-configured model instances
def get_common_ollama_models(api_url: str = "http://127.0.0.1:11434") -> Dict[str, OllamaModel]:
    """Get common Ollama model configurations"""
    return {
        "llama3.2-vision": OllamaModel("llama3.2-vision:latest", api_url),
        "llama3.2-3b": OllamaModel("llama3.2:3b", api_url),
        "gemma2-2b": OllamaModel("gemma2:2b", api_url),
        "gemma2-9b": OllamaModel("gemma2:9b", api_url),
        "mistral": OllamaModel("mistral:latest", api_url),
        "codellama": OllamaModel("codellama:latest", api_url),
    }


# Example usage functions
def test_ollama_connection(api_url: str = "http://127.0.0.1:11434") -> bool:
    """Test connection to Ollama server"""
    try:
        response = requests.get(f"{api_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        logger.info(f"Ollama server accessible. Available models: {len(models)}")
        return True
    except Exception as e:
        logger.error(f"Cannot connect to Ollama server: {str(e)}")
        return False