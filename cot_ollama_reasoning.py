"""
Chain-of-Thought analysis optimized for Ollama phi4-reasoning and other reasoning models
"""

import numpy as np
import logging
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from config import TokenSHAPConfig
# Create simple Ollama interface (self-contained)
import requests
import json

class SimpleOllamaModel:
    """Simple Ollama model interface"""
    def __init__(self, model_name: str, api_url: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.api_url = api_url
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from Ollama model"""
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")

logger = logging.getLogger(__name__)


class OllamaCoTAnalyzer:
    """
    Chain-of-Thought analyzer specifically designed for Ollama reasoning models like phi4-reasoning
    """
    
    def __init__(self, 
                 model_name: str = "phi4-reasoning:latest",
                 api_url: str = "http://127.0.0.1:11434",
                 config: TokenSHAPConfig = None):
        
        self.config = config or TokenSHAPConfig()
        self.model_name = model_name
        self.api_url = api_url
        
        # Initialize Ollama model for reasoning
        self.reasoning_model = SimpleOllamaModel(model_name, api_url)
        
        # Initialize basic TokenSHAP components (simplified for self-contained version)
        from token_shap import EnhancedTokenSHAP
        from sfa_learner import SFAMetaLearner
        
        # Use core components without transformers dependency
        self.token_explainer = None  # Will be initialized if needed
        self.sfa_learner = SFAMetaLearner(config)
        
        # CoT-specific patterns for reasoning models
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        logger.info(f"Initialized CoT Analyzer for {model_name}")
    
    def _initialize_reasoning_patterns(self) -> Dict[str, str]:
        """Initialize reasoning patterns optimized for reasoning models like phi4"""
        return {
            'analytical': "Think through this step by step:",
            'mathematical': "Solve this step by step, showing your work:",
            'logical': "Let's reason through this logically:",
            'comparative': "Compare and analyze the following:",
            'problem_solving': "Break down this problem into steps:",
            'critical_thinking': "Critically analyze this by examining:",
            'deductive': "Use deductive reasoning to conclude:",
            'inductive': "Use inductive reasoning to generalize:",
            'systematic': "Approach this systematically by:",
            'phi4_style': "Let me think about this carefully..."  # Phi4 tends to use this pattern
        }
    
    def generate_reasoning(self, 
                         prompt: str, 
                         pattern: str = 'phi4_style',
                         max_length: int = None) -> Tuple[List[str], str, Dict[str, Any]]:
        """
        Generate Chain-of-Thought reasoning using phi4-reasoning or similar models
        
        Returns:
            Tuple of (reasoning_steps, full_response, metadata)
        """
        if max_length is None:
            max_length = self.config.max_output_length * 3  # Allow longer for reasoning
        
        # Create reasoning prompt
        reasoning_template = self.reasoning_patterns.get(pattern, self.reasoning_patterns['phi4_style'])
        
        # Format prompt for reasoning models
        reasoning_prompt = f"{reasoning_template}\n\n{prompt}\n\nLet me work through this:"
        
        try:
            # Generate reasoning response
            full_response = self.reasoning_model.generate(
                prompt=reasoning_prompt,
                max_length=max_length,
                temperature=0.7  # Good balance for reasoning
            )
            
            # Parse reasoning steps
            reasoning_steps = self._parse_reasoning_steps(full_response, prompt)
            
            # Extract metadata
            metadata = self._extract_reasoning_metadata(full_response, reasoning_steps)
            
            logger.info(f"Generated {len(reasoning_steps)} reasoning steps")
            
            return reasoning_steps, full_response, metadata
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}")
            return [], "", {'error': str(e)}
    
    def _parse_reasoning_steps(self, response: str, original_prompt: str) -> List[str]:
        """Parse reasoning response into individual steps - optimized for reasoning models"""
        
        # Remove original prompt if present
        clean_response = response
        if original_prompt in clean_response:
            clean_response = clean_response.replace(original_prompt, '').strip()
        
        # Patterns that reasoning models commonly use
        step_patterns = [
            r'Step \d+:',           # Step 1:, Step 2:, etc.
            r'\d+\.',               # 1., 2., 3., etc.
            r'First[,:]',           # First:, First,
            r'Second[,:]',          # Second:, Second,
            r'Third[,:]',           # Third:, Third,
            r'Next[,:]',            # Next:, Next,
            r'Then[,:]',            # Then:, Then,
            r'Finally[,:]',         # Finally:, Finally,
            r'Therefore[,:]',       # Therefore:, Therefore,
            r'So[,:]',              # So:, So,
            r'Let me[,:]',          # Let me:, Let me,
            r'I need to[,:]',       # I need to:, I need to,
            r'Looking at[,:]',      # Looking at:, Looking at,
            r'Considering[,:]',     # Considering:, Considering,
        ]
        
        # Split by double newlines first (paragraph breaks)
        paragraphs = [p.strip() for p in clean_response.split('\n\n') if p.strip()]
        
        steps = []
        
        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.split('\n') if line.strip()]
            
            current_step = []
            
            for line in lines:
                # Check if this line starts a new reasoning step
                is_new_step = any(re.match(pattern, line, re.IGNORECASE) for pattern in step_patterns)
                
                if is_new_step and current_step:
                    # Save current step and start new one
                    step_text = ' '.join(current_step).strip()
                    if len(step_text) > 15:  # Filter out very short steps
                        steps.append(step_text)
                    current_step = [line]
                else:
                    current_step.append(line)
            
            # Don't forget the last step in this paragraph
            if current_step:
                step_text = ' '.join(current_step).strip()
                if len(step_text) > 15:
                    steps.append(step_text)
        
        # If no clear steps found, try to split by sentences
        if not steps and clean_response:
            sentences = re.split(r'[.!?]+', clean_response)
            steps = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Limit to reasonable number of steps
        max_steps = self.config.cot_max_steps
        if len(steps) > max_steps:
            steps = steps[:max_steps]
        
        return steps
    
    def _extract_reasoning_metadata(self, response: str, steps: List[str]) -> Dict[str, Any]:
        """Extract metadata about the reasoning process"""
        
        metadata = {
            'total_length': len(response),
            'num_steps': len(steps),
            'avg_step_length': np.mean([len(step) for step in steps]) if steps else 0,
            'contains_math': bool(re.search(r'[\d\+\-\*\/\=\(\)]', response)),
            'contains_logic': bool(re.search(r'\b(if|then|therefore|because|since|thus|hence)\b', response, re.IGNORECASE)),
            'confidence_indicators': [],
            'reasoning_quality': 0.0
        }
        
        # Look for confidence indicators
        confidence_patterns = [
            r'\b(certainly|definitely|clearly|obviously)\b',
            r'\b(probably|likely|possibly|maybe)\b',
            r'\b(I think|I believe|I suspect)\b',
            r'\b(uncertain|unsure|unclear)\b'
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                metadata['confidence_indicators'].extend(matches)
        
        # Simple reasoning quality score based on structure
        quality_score = 0.0
        if len(steps) >= 2:
            quality_score += 0.3  # Multi-step reasoning
        if metadata['contains_logic']:
            quality_score += 0.3  # Uses logical connectives
        if len(steps) >= 3:
            quality_score += 0.2  # Good depth
        if metadata['avg_step_length'] > 30:
            quality_score += 0.2  # Detailed steps
        
        metadata['reasoning_quality'] = min(quality_score, 1.0)
        
        return metadata
    
    def analyze_cot_attribution(self, 
                               prompt: str, 
                               pattern: str = 'phi4_style',
                               analyze_steps: bool = True) -> Dict[str, Any]:
        """
        Complete Chain-of-Thought attribution analysis
        
        Returns comprehensive analysis including:
        1. Reasoning steps generation
        2. Step-level importance 
        3. Token-level attribution within steps
        4. Overall reasoning quality assessment
        """
        
        logger.info(f"Starting CoT attribution analysis for prompt: '{prompt[:50]}...'")
        
        # Step 1: Generate reasoning
        reasoning_steps, full_response, metadata = self.generate_reasoning(prompt, pattern)
        
        if not reasoning_steps:
            return {
                'error': 'no_reasoning_generated',
                'prompt': prompt,
                'response': full_response,
                'metadata': metadata
            }
        
        # Step 2: Analyze step importance
        step_importance = self._compute_step_importance(reasoning_steps, full_response)
        
        # Step 3: Token-level analysis of each step (if requested)
        token_attributions = []
        step_complexities = []
        
        if analyze_steps:
            logger.info("Analyzing token-level attributions for each step...")
            
            for i, step in enumerate(reasoning_steps):
                try:
                    # Use a lightweight config for step analysis
                    step_config = TokenSHAPConfig(max_samples=5)  # Fast analysis
                    step_result = self.token_explainer.explain(step)
                    
                    token_attributions.append(step_result)
                    
                    # Calculate step complexity
                    complexity = sum(abs(v) for v in step_result.values())
                    step_complexities.append(complexity)
                    
                    logger.info(f"Analyzed step {i+1}/{len(reasoning_steps)}")
                    
                except Exception as e:
                    logger.warning(f"Error analyzing step {i+1}: {str(e)}")
                    token_attributions.append({})
                    step_complexities.append(0.0)
        
        # Step 4: Identify critical components
        critical_steps = self._identify_critical_steps(reasoning_steps, step_importance)
        
        if token_attributions:
            critical_tokens = self._identify_critical_tokens(token_attributions, reasoning_steps)
        else:
            critical_tokens = []
        
        # Step 5: Compute overall metrics
        reasoning_depth = len(reasoning_steps)
        avg_step_complexity = np.mean(step_complexities) if step_complexities else 0
        chain_coherence = self._compute_chain_coherence(reasoning_steps)
        
        # Compile comprehensive result
        result = {
            'prompt': prompt,
            'reasoning_pattern': pattern,
            'reasoning_steps': reasoning_steps,
            'full_response': full_response,
            'metadata': metadata,
            'step_importance': step_importance,
            'token_attributions': token_attributions,
            'step_complexities': step_complexities,
            'critical_steps': critical_steps,
            'critical_tokens': critical_tokens,
            'metrics': {
                'reasoning_depth': reasoning_depth,
                'avg_step_complexity': avg_step_complexity,
                'chain_coherence': chain_coherence,
                'reasoning_quality': metadata.get('reasoning_quality', 0.0),
                'total_tokens_analyzed': sum(len(attr) for attr in token_attributions)
            }
        }
        
        logger.info(f"CoT analysis complete: {reasoning_depth} steps, quality={metadata.get('reasoning_quality', 0.0):.2f}")
        
        return result
    
    def _compute_step_importance(self, steps: List[str], full_response: str) -> List[float]:
        """Compute importance of each reasoning step"""
        if not steps:
            return []
        
        importance_scores = []
        
        # Simple heuristic-based importance scoring for reasoning steps
        for i, step in enumerate(steps):
            score = 0.0
            
            # Position-based scoring
            if i == 0:  # First step often sets up the problem
                score += 0.2
            if i == len(steps) - 1:  # Last step often contains conclusion
                score += 0.3
            
            # Content-based scoring
            step_lower = step.lower()
            
            # Logic indicators
            if any(word in step_lower for word in ['therefore', 'thus', 'hence', 'conclusion']):
                score += 0.4
            if any(word in step_lower for word in ['because', 'since', 'given that']):
                score += 0.3
            if any(word in step_lower for word in ['if', 'then', 'when', 'suppose']):
                score += 0.2
            
            # Question/problem indicators
            if '?' in step:
                score += 0.2
            
            # Mathematical content
            if re.search(r'[\d\+\-\*\/\=]', step):
                score += 0.2
            
            # Length-based (longer steps often more important)
            if len(step) > 100:
                score += 0.1
            
            importance_scores.append(min(score, 1.0))  # Cap at 1.0
        
        # Normalize to sum to 1
        total = sum(importance_scores)
        if total > 0:
            importance_scores = [s / total for s in importance_scores]
        else:
            importance_scores = [1.0 / len(steps)] * len(steps)
        
        return importance_scores
    
    def _compute_chain_coherence(self, steps: List[str]) -> float:
        """Compute coherence of the reasoning chain"""
        if len(steps) <= 1:
            return 1.0
        
        # Simple word overlap coherence between consecutive steps
        coherence_scores = []
        
        for i in range(len(steps) - 1):
            words1 = set(steps[i].lower().split())
            words2 = set(steps[i + 1].lower().split())
            
            if not words1 or not words2:
                coherence_scores.append(0.0)
                continue
            
            # Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            coherence = intersection / union if union > 0 else 0.0
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _identify_critical_steps(self, steps: List[str], importance: List[float], threshold: float = 0.15) -> List[Dict[str, Any]]:
        """Identify the most critical reasoning steps"""
        critical = []
        
        for i, (step, imp) in enumerate(zip(steps, importance)):
            if imp > threshold:
                critical.append({
                    'index': i,
                    'step_preview': step[:100] + '...' if len(step) > 100 else step,
                    'full_step': step,
                    'importance': float(imp),
                    'rank': len([s for s in importance if s > imp]) + 1
                })
        
        return sorted(critical, key=lambda x: x['importance'], reverse=True)
    
    def _identify_critical_tokens(self, attributions: List[Dict[str, float]], steps: List[str], top_k: int = 15) -> List[Dict[str, Any]]:
        """Identify most critical tokens across all reasoning steps"""
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
    
    def visualize_reasoning_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """Create a text-based visualization of the reasoning analysis"""
        
        if 'error' in analysis_result:
            return f"❌ Analysis Error: {analysis_result['error']}"
        
        viz = []
        viz.append("🧠 Chain-of-Thought Reasoning Analysis")
        viz.append("=" * 50)
        
        # Basic info
        metrics = analysis_result.get('metrics', {})
        viz.append(f"📊 Reasoning Depth: {metrics.get('reasoning_depth', 0)} steps")
        viz.append(f"🎯 Reasoning Quality: {metrics.get('reasoning_quality', 0.0):.2f}/1.0")
        viz.append(f"🔗 Chain Coherence: {metrics.get('chain_coherence', 0.0):.2f}")
        viz.append("")
        
        # Critical steps
        critical_steps = analysis_result.get('critical_steps', [])
        if critical_steps:
            viz.append("🎯 Most Critical Reasoning Steps:")
            for i, step_info in enumerate(critical_steps[:3], 1):
                importance = step_info['importance']
                preview = step_info['step_preview']
                viz.append(f"  {i}. [{importance:.3f}] {preview}")
            viz.append("")
        
        # Step-by-step breakdown
        steps = analysis_result.get('reasoning_steps', [])
        importance = analysis_result.get('step_importance', [])
        
        if steps:
            viz.append("📝 Step-by-Step Analysis:")
            for i, (step, imp) in enumerate(zip(steps, importance), 1):
                # Create importance bar
                bar_length = int(imp * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                preview = step[:80] + "..." if len(step) > 80 else step
                viz.append(f"  Step {i}: |{bar}| {imp:.3f}")
                viz.append(f"    {preview}")
                viz.append("")
        
        # Critical tokens
        critical_tokens = analysis_result.get('critical_tokens', [])
        if critical_tokens:
            viz.append("🔤 Most Important Tokens Across All Steps:")
            for i, token_info in enumerate(critical_tokens[:10], 1):
                token = token_info['token']
                value = token_info['shapley_value']
                step_idx = token_info['step_index']
                viz.append(f"  {i:2d}. '{token:<12}' → {value:+.4f} (Step {step_idx + 1})")
            viz.append("")
        
        return '\n'.join(viz)


# Convenience function for quick analysis
def quick_cot_analysis(prompt: str, 
                      model_name: str = "phi4-reasoning:latest",
                      api_url: str = "http://127.0.0.1:11434") -> str:
    """
    Quick CoT analysis with text visualization
    """
    try:
        # Create a proper config for the analyzer
        config = TokenSHAPConfig(
            max_samples=10,        # Fast analysis
            parallel_workers=1,    # Single worker for simplicity
            cot_max_steps=5       # Limit steps for quick analysis
        )
        analyzer = OllamaCoTAnalyzer(model_name, api_url, config)
        result = analyzer.analyze_cot_attribution(prompt, analyze_steps=False)  # Skip token analysis for speed
        return analyzer.visualize_reasoning_analysis(result)
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"


# Example usage
if __name__ == "__main__":
    print("Ollama CoT Reasoning Analysis")
    print("=" * 40)
    
    # Test with phi4-reasoning (adjust model name if different)
    test_prompt = "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?"
    
    result = quick_cot_analysis(test_prompt, model_name="gemma3:270m")  # Using available model for demo
    print(result)