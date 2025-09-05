"""
Configuration and enums for TokenSHAP with SFA
"""

from dataclasses import dataclass
from enum import Enum
import torch


class AttributionMethod(Enum):
    """Attribution method types"""
    TOKENSHAP = "tokenshap"
    SFA = "sfa"
    HYBRID = "hybrid"
    COT_HIERARCHICAL = "cot_hierarchical"


@dataclass
class TokenSHAPConfig:
    """Enhanced configuration for TokenSHAP with SFA"""
    # Sampling parameters
    max_samples: int = 100
    min_samples: int = 20
    batch_size: int = 10
    
    # Convergence parameters
    convergence_threshold: float = 0.01
    convergence_checks: int = 3
    adaptive_convergence: bool = True
    early_stopping: bool = True
    
    # Stratification parameters
    use_stratification: bool = True
    ensure_first_order: bool = True
    k_folds: int = 5
    
    # Performance parameters
    cache_responses: bool = True
    cache_size: int = 10000
    parallel_workers: int = 4
    use_gpu: bool = torch.cuda.is_available()
    
    # Model parameters
    max_input_length: int = 512
    max_output_length: int = 256
    temperature: float = 0.7
    
    # SFA parameters
    sfa_n_estimators: int = 100
    sfa_max_depth: int = 10
    sfa_min_samples_train: int = 10
    
    # CoT parameters
    cot_max_steps: int = 10
    cot_step_delimiter: str = "\n"
    cot_prompt_template: str = "Let's think step by step:\n"