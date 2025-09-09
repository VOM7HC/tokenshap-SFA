"""
Chain-of-Thought aware hierarchical TokenSHAP with SFA
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, PreTrainedModel
from config import TokenSHAPConfig
from utils import TokenProcessor
from value_functions import SimilarityValueFunction
from token_shap import EnhancedTokenSHAP
from sfa_learner import SFAMetaLearner

logger = logging.getLogger(__name__)


class CoTTokenSHAP:
    """
    Chain-of-Thought aware hierarchical TokenSHAP with SFA
    """
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: AutoTokenizer,
                 config: TokenSHAPConfig = None):
        self.config = config or TokenSHAPConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = TokenProcessor(tokenizer)
        
        # Initialize base explainer
        self.token_explainer = EnhancedTokenSHAP(model, tokenizer, config)
        self.sfa_learner = SFAMetaLearner(config)
        
        # CoT-specific components
        self.step_value_function = SimilarityValueFunction()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
    
    def _initialize_reasoning_patterns(self) -> Dict[str, str]:
        """Initialize common reasoning patterns for CoT"""
        return {
            'analytical': "Let's analyze this step by step:\n",
            'mathematical': "Let's solve this mathematically:\n",
            'logical': "Let's think through this logically:\n",
            'comparative': "Let's compare the options:\n",
            'sequential': "Let's go through this sequentially:\n"
        }
    
    def generate_cot(self, 
                    prompt: str, 
                    pattern: str = 'analytical') -> Tuple[List[str], str]:
        """
        Generate Chain-of-Thought reasoning steps
        
        Returns:
            Tuple of (steps_list, full_response)
        """
        # Add CoT prompt template
        cot_template = self.reasoning_patterns.get(pattern, self.config.cot_prompt_template)
        cot_prompt = f"{prompt}\n{cot_template}"
        
        # Generate response
        inputs = self.processor.encode(cot_prompt, self.config.max_input_length)
        
        if self.config.use_gpu and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_output_length * 2,  # Allow longer for CoT
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95
            )
        
        full_response = self.processor.decode(outputs)
        
        # Parse steps
        steps = self._parse_cot_steps(full_response, prompt)
        
        return steps, full_response
    
    def _parse_cot_steps(self, response: str, original_prompt: str) -> List[str]:
        """Parse CoT response into individual reasoning steps"""
        # Remove original prompt if present
        if original_prompt in response:
            response = response.replace(original_prompt, '').strip()
        
        # Split by various delimiters
        potential_delimiters = [
            self.config.cot_step_delimiter,
            '\n\n',
            'Step ',
            '- ',
            '• ',
            'First,',
            'Second,',
            'Third,',
            'Finally,',
            'Therefore,',
            'Thus,'
        ]
        
        steps = []
        current_step = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts a new step
            is_new_step = any(line.startswith(delim.strip()) for delim in potential_delimiters[2:])
            
            if is_new_step and current_step:
                steps.append(' '.join(current_step))
                current_step = [line]
            else:
                current_step.append(line)
        
        # Add last step
        if current_step:
            steps.append(' '.join(current_step))
        
        # Filter and clean steps
        steps = [s.strip() for s in steps if s.strip() and len(s.strip()) > 10]
        
        # Limit to max steps
        return steps[:self.config.cot_max_steps]
    
    def compute_hierarchical_attribution(self, 
                                        prompt: str,
                                        use_sfa: bool = True,
                                        pattern: str = 'analytical') -> Dict[str, Any]:
        """
        Compute hierarchical attribution for CoT reasoning
        
        Three levels of attribution:
        1. Token-level: Within each reasoning step
        2. Step-level: Importance of each step
        3. Chain-level: Overall reasoning quality
        """
        # Generate CoT
        cot_steps, full_response = self.generate_cot(prompt, pattern)
        
        if not cot_steps:
            logger.warning("No CoT steps generated, falling back to direct attribution")
            return {
                'error': 'no_cot_steps',
                'direct_attribution': self.token_explainer.compute_shapley_values(prompt)
            }
        
        # Level 1: Token-level attribution for each step
        token_attributions = []
        step_complexities = []
        
        for step_idx, step in enumerate(cot_steps):
            if use_sfa and self.sfa_learner.is_trained:
                # Use SFA for faster prediction
                tokens = self.processor.tokenize(step)
                token_shapley = self.sfa_learner.predict(step, tokens)
            else:
                # Use full TokenSHAP computation
                token_shapley = self.token_explainer.compute_shapley_values(step)
            
            token_attributions.append(token_shapley)
            
            # Compute step complexity (sum of absolute Shapley values)
            complexity = sum(abs(v) for v in token_shapley.values())
            step_complexities.append(complexity)
        
        # Level 2: Step-level importance
        step_importance = self._compute_step_importance(cot_steps, full_response)
        
        # Level 3: Chain-level coherence
        chain_coherence = self._compute_chain_coherence(cot_steps, step_importance)
        
        # Identify critical components
        critical_steps = self._identify_critical_steps(
            cot_steps, step_importance, threshold=0.15
        )
        critical_tokens = self._identify_critical_tokens(
            token_attributions, cot_steps, top_k=20
        )
        
        # Compute aggregated metrics
        avg_step_complexity = np.mean(step_complexities) if step_complexities else 0
        reasoning_depth = len(cot_steps)
        
        return {
            'prompt': prompt,
            'cot_steps': cot_steps,
            'token_attributions': token_attributions,
            'step_importance': step_importance,
            'step_complexities': step_complexities,
            'chain_coherence': chain_coherence,
            'critical_steps': critical_steps,
            'critical_tokens': critical_tokens,
            'metrics': {
                'reasoning_depth': reasoning_depth,
                'avg_step_complexity': avg_step_complexity,
                'chain_coherence': chain_coherence,
                'total_tokens_analyzed': sum(len(attr) for attr in token_attributions)
            }
        }

    def compute_hierarchical_attribution_augmented(self, 
                                                 prompt: str,
                                                 sfa_learner: 'SFAMetaLearner') -> Dict[str, Any]:
        """
        Enhanced CoT attribution using trained SFA for acceleration
        """
        # Generate CoT steps
        cot_steps, full_response = self.generate_cot(prompt)
        
        if not cot_steps:
            return {'error': 'no_cot_steps'}
        
        # Use SFA for fast token attribution on each step
        token_attributions = []
        step_complexities = []
        
        for step in cot_steps:
            tokens = self.processor.tokenize(step)
            
            # Use SFA predictions augmented with initial Shapley estimates
            if sfa_learner.is_trained:
                # Get quick SFA prediction
                sfa_prediction = sfa_learner.predict(step, tokens)
                
                # Use augmented prediction for better accuracy
                augmented_attribution = sfa_learner.predict_augmented(
                    step, tokens, sfa_prediction
                )
                token_attributions.append(augmented_attribution)
            else:
                # Fallback to standard Shapley
                token_attributions.append(
                    self.token_explainer.compute_shapley_values(step)
                )
            
            complexity = sum(abs(v) for v in token_attributions[-1].values())
            step_complexities.append(complexity)
        
        # Continue with rest of hierarchical analysis...
        step_importance = self._compute_step_importance(cot_steps, full_response)
        chain_coherence = self._compute_chain_coherence(cot_steps, step_importance)
        
        return {
            'prompt': prompt,
            'cot_steps': cot_steps,
            'token_attributions': token_attributions,
            'step_importance': step_importance,
            'step_complexities': step_complexities,
            'chain_coherence': chain_coherence,
            'augmented': True  # Flag to indicate SFA augmentation was used
        }
    
    def _compute_step_importance(self, steps: List[str], full_response: str) -> List[float]:
        """Compute importance of each reasoning step"""
        if not steps:
            return []
        
        importance_scores = []
        
        for i, step in enumerate(steps):
            # Create response without this step
            other_steps = steps[:i] + steps[i+1:]
            partial_response = ' '.join(other_steps)
            
            # Compute importance as drop in value
            full_value = self.step_value_function.compute(full_response, full_response)
            partial_value = self.step_value_function.compute(full_response, partial_response)
            
            importance = full_value - partial_value
            importance_scores.append(max(0, importance))  # Ensure non-negative
        
        # Normalize
        total = sum(importance_scores)
        if total > 0:
            importance_scores = [s / total for s in importance_scores]
        else:
            # Equal importance if no variation
            importance_scores = [1.0 / len(steps)] * len(steps)
        
        return importance_scores
    
    def _compute_chain_coherence(self, steps: List[str], importance: List[float]) -> float:
        """Compute coherence score for the reasoning chain"""
        if len(steps) <= 1:
            return 1.0
        
        # Compute pairwise coherence between consecutive steps
        coherence_scores = []
        
        for i in range(len(steps) - 1):
            coherence = self.step_value_function.compute(steps[i], steps[i+1])
            # Weight by importance of both steps
            weighted_coherence = coherence * (importance[i] + importance[i+1]) / 2
            coherence_scores.append(weighted_coherence)
        
        # Average coherence
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _identify_critical_steps(self, 
                                steps: List[str], 
                                importance: List[float],
                                threshold: float = 0.15) -> List[Dict[str, Any]]:
        """Identify critical reasoning steps"""
        critical = []
        
        for idx, (step, imp) in enumerate(zip(steps, importance)):
            if imp > threshold:
                critical.append({
                    'index': idx,
                    'step': step[:100] + '...' if len(step) > 100 else step,
                    'importance': float(imp),
                    'rank': len([i for i in importance if i > imp]) + 1
                })
        
        return sorted(critical, key=lambda x: x['importance'], reverse=True)
    
    def _identify_critical_tokens(self,
                                 attributions: List[Dict[str, float]],
                                 steps: List[str],
                                 top_k: int = 20) -> List[Dict[str, Any]]:
        """Identify most critical tokens across all steps"""
        all_tokens = []
        
        for step_idx, (attr_dict, step) in enumerate(zip(attributions, steps)):
            for token, value in attr_dict.items():
                all_tokens.append({
                    'token': token,
                    'shapley_value': float(value),
                    'abs_value': abs(float(value)),
                    'step_index': step_idx,
                    'step_preview': step[:50] + '...' if len(step) > 50 else step
                })
        
        # Sort by absolute Shapley value
        all_tokens.sort(key=lambda x: x['abs_value'], reverse=True)
        
        return all_tokens[:top_k]
    
    def train_sfa_on_cot(self, 
                        prompts: List[str],
                        patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train SFA specifically on CoT-generated explanations"""
        training_data = []
        
        if patterns is None:
            patterns = ['analytical'] * len(prompts)
        
        logger.info(f"Training SFA on {len(prompts)} CoT examples...")
        
        for prompt, pattern in zip(prompts, patterns):
            # Generate CoT and compute attributions
            cot_steps, _ = self.generate_cot(prompt, pattern)
            
            for step in cot_steps:
                shapley_values = self.token_explainer.compute_shapley_values(step)
                training_data.append((step, shapley_values))
        
        # Train SFA
        training_result = self.sfa_learner.train(training_data)
        
        return {
            'n_prompts': len(prompts),
            'n_steps': len(training_data),
            'training_result': training_result
        }