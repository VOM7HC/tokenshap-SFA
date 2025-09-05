"""
Enhanced TokenSHAP with SFA (Shapley-based Feature Augmentation) and CoT Integration v2.0
Improved implementation with bug fixes and architectural enhancements
"""

# Main imports for easy access
from .tokenshap_with_sfa import TokenSHAPWithSFA
from .tokenshap_ollama import TokenSHAPWithOllama, OllamaTokenSHAP
from .config import TokenSHAPConfig, AttributionMethod
from .token_shap import EnhancedTokenSHAP
from .sfa_learner import SFAMetaLearner
from .cot_explainer import CoTTokenSHAP
from .value_functions import ValueFunction, SimilarityValueFunction, ProbabilityValueFunction
from .utils import ThreadSafeCache, TokenProcessor
from .ollama_integration import OllamaModel, SimpleOllamaModel, create_ollama_model

__version__ = "2.0.0"
__author__ = "TokenSHAP Team"

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Export main classes and functions
__all__ = [
    'TokenSHAPWithSFA',
    'TokenSHAPWithOllama',
    'OllamaTokenSHAP',
    'TokenSHAPConfig', 
    'AttributionMethod',
    'EnhancedTokenSHAP',
    'SFAMetaLearner',
    'CoTTokenSHAP',
    'ValueFunction',
    'SimilarityValueFunction', 
    'ProbabilityValueFunction',
    'ThreadSafeCache',
    'TokenProcessor',
    'OllamaModel',
    'SimpleOllamaModel',
    'create_ollama_model'
]