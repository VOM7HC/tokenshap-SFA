"""
Enhanced TokenSHAP with SFA (Shapley-based Feature Augmentation) and CoT Integration v2.0
Improved implementation with bug fixes and architectural enhancements
"""

# Main imports for easy access
from .tokenshap_with_sfa import TokenSHAPWithSFA
from .config import TokenSHAPConfig, AttributionMethod
from .token_shap import EnhancedTokenSHAP
from .sfa_learner import SFAMetaLearner
from .cot_explainer import CoTTokenSHAP
from .value_functions import ValueFunction, SimilarityValueFunction, ProbabilityValueFunction
from .utils import ThreadSafeCache, TokenProcessor

__version__ = "2.0.0"
__author__ = "TokenSHAP Team"

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Export main classes and functions
__all__ = [
    'TokenSHAPWithSFA',
    'TokenSHAPConfig', 
    'AttributionMethod',
    'EnhancedTokenSHAP',
    'SFAMetaLearner',
    'CoTTokenSHAP',
    'ValueFunction',
    'SimilarityValueFunction', 
    'ProbabilityValueFunction',
    'ThreadSafeCache',
    'TokenProcessor'
]