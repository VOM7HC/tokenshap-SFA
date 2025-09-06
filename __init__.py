"""
TokenSHAP with SFA (Shapley-based Feature Augmentation) - Core ML/DL Algorithm
Streamlined implementation focused on core ML algorithms and essential functionality
"""

# Core ML algorithm imports
from .tokenshap_with_sfa import TokenSHAPWithSFA
from .cot_ollama_reasoning import OllamaCoTAnalyzer, quick_cot_analysis
from .config import TokenSHAPConfig, AttributionMethod
from .token_shap import EnhancedTokenSHAP
from .sfa_learner import SFAMetaLearner
from .cot_explainer import CoTTokenSHAP
from .value_functions import ValueFunction, SimilarityValueFunction, ProbabilityValueFunction
from .utils import ThreadSafeCache, TokenProcessor

__version__ = "3.0.0"
__author__ = "TokenSHAP Team"

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Export core ML/DL components
__all__ = [
    # Core ML Algorithm
    'TokenSHAPWithSFA',
    'EnhancedTokenSHAP',
    'SFAMetaLearner',
    
    # Chain-of-Thought ML
    'CoTTokenSHAP',
    'OllamaCoTAnalyzer',
    'quick_cot_analysis',
    
    # Configuration and Methods
    'TokenSHAPConfig', 
    'AttributionMethod',
    
    # Value Functions (ML Components)
    'ValueFunction',
    'SimilarityValueFunction', 
    'ProbabilityValueFunction',
    
    # Utilities
    'ThreadSafeCache',
    'TokenProcessor'
]