"""
SFA (Shapley-based Feature Augmentation) Meta-Learner
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from config import TokenSHAPConfig

logger = logging.getLogger(__name__)


class SFAMetaLearner:
    """
    Enhanced Shapley-based Feature Augmentation Meta-Learner
    """
    
    def __init__(self, config: TokenSHAPConfig):
        self.config = config
        self.meta_model = GradientBoostingRegressor(
            n_estimators=config.sfa_n_estimators,
            max_depth=config.sfa_max_depth,
            random_state=42
        )
        self.feature_vectorizer = TfidfVectorizer(max_features=100)
        self.is_trained = False
        self.feature_names = []
        self.training_history = []
    
    def extract_features(self, tokens: List[str], context: str) -> np.ndarray:
        """Extract comprehensive features for tokens"""
        features = []
        n_tokens = len(tokens)
        
        # Precompute context features
        context_lower = context.lower()
        context_words = set(context_lower.split())
        
        for i, token in enumerate(tokens):
            token_features = []
            token_lower = token.lower()
            
            # Lexical features
            token_features.extend([
                len(token),                                    # Length
                1 if token.isalpha() else 0,                  # Is alphabetic
                1 if token.isdigit() else 0,                  # Is numeric
                1 if token.isupper() else 0,                  # Is uppercase
                1 if token.islower() else 0,                  # Is lowercase
                1 if token.istitle() else 0,                  # Is title case
                1 if any(c.isdigit() for c in token) else 0, # Contains digits
                token.count('-'),                              # Hyphen count
                1 if token.startswith('##') else 0,           # Is subword token
            ])
            
            # Position features
            relative_position = i / max(1, n_tokens - 1) if n_tokens > 1 else 0.5
            token_features.extend([
                relative_position,                    # Relative position
                1 if i == 0 else 0,                  # Is first
                1 if i == n_tokens - 1 else 0,      # Is last
                1 if i < n_tokens // 3 else 0,      # In first third
                1 if i > 2 * n_tokens // 3 else 0,  # In last third
                min(i, 10) / 10,                    # Distance from start (normalized)
                min(n_tokens - i - 1, 10) / 10,     # Distance from end (normalized)
            ])
            
            # Context features
            token_features.extend([
                context_lower.count(token_lower),           # Frequency in context
                1 if token_lower in context_words else 0,   # Appears as word
                len(token) / len(context) if context else 0, # Length ratio
            ])
            
            # Neighboring features (if applicable)
            if i > 0:
                prev_token = tokens[i-1]
                token_features.extend([
                    1 if prev_token.isalpha() else 0,
                    len(prev_token) / 10,
                ])
            else:
                token_features.extend([0, 0])
            
            if i < n_tokens - 1:
                next_token = tokens[i+1]
                token_features.extend([
                    1 if next_token.isalpha() else 0,
                    len(next_token) / 10,
                ])
            else:
                token_features.extend([0, 0])
            
            # Linguistic features
            token_features.extend([
                1 if token in {'.', ',', '!', '?', ';', ':'} else 0,  # Is punctuation
                1 if token in {'the', 'a', 'an', 'is', 'are'} else 0, # Is common word
                1 if len(token) > 7 else 0,                            # Is long word
                1 if len(token) <= 2 else 0,                           # Is short word
            ])
            
            features.append(token_features)
        
        return np.array(features, dtype=np.float32)
    
    def train(self, training_data: List[Tuple[str, Dict[str, float]]], 
              use_cv: bool = True) -> Dict[str, float]:
        """
        Train meta-learner with cross-validation
        """
        if len(training_data) < self.config.sfa_min_samples_train:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return {'error': 'insufficient_data'}
        
        # Prepare training data
        X_all = []
        y_all = []
        tokens_all = []
        
        for prompt, shapley_dict in training_data:
            tokens = prompt.split()  # Should use proper tokenization
            features = self.extract_features(tokens, prompt)
            
            for i, token in enumerate(tokens):
                if token in shapley_dict:
                    X_all.append(features[i])
                    y_all.append(shapley_dict[token])
                    tokens_all.append(token)
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # Cross-validation
        if use_cv and len(X_all) >= self.config.k_folds:
            cv_scores = []
            kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X_all):
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                y_train, y_val = y_all[train_idx], y_all[val_idx]
                
                # Train fold model
                fold_model = GradientBoostingRegressor(
                    n_estimators=self.config.sfa_n_estimators,
                    max_depth=self.config.sfa_max_depth,
                    random_state=42
                )
                fold_model.fit(X_train, y_train)
                
                # Validate
                val_pred = fold_model.predict(X_val)
                mse = np.mean((val_pred - y_val) ** 2)
                cv_scores.append(mse)
            
            avg_cv_score = np.mean(cv_scores)
            logger.info(f"Cross-validation MSE: {avg_cv_score:.4f}")
        
        # Train final model on all data
        self.meta_model.fit(X_all, y_all)
        self.is_trained = True
        
        # Store feature names for interpretability
        self.feature_names = [
            'length', 'is_alpha', 'is_digit', 'is_upper', 'is_lower', 'is_title',
            'contains_digit', 'hyphen_count', 'is_subword', 'relative_pos',
            'is_first', 'is_last', 'in_first_third', 'in_last_third',
            'dist_from_start', 'dist_from_end', 'freq_in_context', 'appears_as_word',
            'length_ratio', 'prev_is_alpha', 'prev_length', 'next_is_alpha',
            'next_length', 'is_punctuation', 'is_common', 'is_long', 'is_short'
        ]
        
        # Store training history
        self.training_history.append({
            'n_samples': len(X_all),
            'n_prompts': len(training_data),
            'cv_score': avg_cv_score if use_cv else None
        })
        
        return {
            'n_samples': len(X_all),
            'cv_score': avg_cv_score if use_cv else None,
            'feature_importance': self.get_feature_importance()
        }
    
    def predict(self, prompt: str, tokens: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict Shapley values for new prompt"""
        if not self.is_trained:
            raise ValueError("Meta-learner not trained yet")
        
        if tokens is None:
            tokens = prompt.split()  # Should use proper tokenization
        
        features = self.extract_features(tokens, prompt)
        predictions = self.meta_model.predict(features)
        
        return {token: float(predictions[i]) for i, token in enumerate(tokens)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {}
        
        importance = self.meta_model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}