"""
SFA (Shapley-based Feature Augmentation) Meta-Learner - Enhanced Version
Enhanced version with proper feature augmentation and dual-stage training
"""

import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from config import TokenSHAPConfig

logger = logging.getLogger(__name__)


class SFAMetaLearner:
    """Enhanced SFA Meta-Learner with proper feature augmentation"""
    
    def __init__(self, config: TokenSHAPConfig):
        self.config = config
        self.meta_model = None
        self.augmentation_model = None  # Additional model for augmented features
        self.feature_vectorizer = TfidfVectorizer(max_features=100)
        self.is_trained = False
        self.feature_names = []
        self.training_history = []
        self.shapley_cache = {}  # Cache computed Shapley values
        self.augmented_features_cache = {}
        
        logger.info(f"Initialized enhanced SFA with {config.sfa_n_estimators} estimators")
    
    def extract_features(self, tokens: List[str], context: str) -> np.ndarray:
        """Extract base features from tokens and context"""
        features = []
        
        for i, token in enumerate(tokens):
            token_features = []
            
            # Basic token features
            token_features.append(len(token))  # Token length
            token_features.append(i / len(tokens))  # Relative position
            token_features.append(token.count('_'))  # Special character count
            token_features.append(float(token.isdigit()))  # Is numeric
            token_features.append(float(token.isupper()))  # Is uppercase
            token_features.append(float(token.islower()))  # Is lowercase
            
            # Context features
            token_features.append(context.count(token))  # Token frequency in context
            token_features.append(float(i == 0))  # Is first token
            token_features.append(float(i == len(tokens) - 1))  # Is last token
            
            # N-gram features
            if i > 0:
                bigram = f"{tokens[i-1]}_{token}"
                token_features.append(context.count(bigram))
            else:
                token_features.append(0)
            
            features.append(token_features)
        
        return np.array(features, dtype=np.float32)
        
    def extract_augmented_features(self, tokens: List[str], context: str, 
                                  shapley_values: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Extract features augmented with Shapley values and predictions"""
        
        # Base features
        base_features = self.extract_features(tokens, context)
        
        if shapley_values is None:
            return base_features
            
        # Augment with Shapley values (SFA core concept)
        augmented = []
        for i, token in enumerate(tokens):
            token_features = base_features[i].tolist()
            
            # Add Shapley value as feature
            token_shapley = shapley_values.get(token, 0.0)
            token_features.append(token_shapley)
            
            # Add Shapley-derived features
            token_features.append(abs(token_shapley))  # Absolute importance
            token_features.append(token_shapley ** 2)  # Squared importance
            token_features.append(np.sign(token_shapley))  # Sign of contribution
            
            # Add interaction features
            if i > 0:
                prev_shapley = shapley_values.get(tokens[i-1], 0.0)
                token_features.append(token_shapley * prev_shapley)  # Interaction
                token_features.append(token_shapley - prev_shapley)  # Difference
            else:
                token_features.extend([0.0, 0.0])
            
            augmented.append(token_features)
            
        return np.array(augmented, dtype=np.float32)
    
    def train(self, training_data: List[Tuple[str, Dict[str, float]]]) -> Dict[str, float]:
        """Train the SFA meta-learner with standard approach (for backward compatibility)"""
        return self.train_with_augmentation(training_data)
    
    def train_with_augmentation(self, training_data: List[Tuple[str, Dict[str, float]]]) -> Dict[str, float]:
        """Train with proper SFA augmentation"""
        
        logger.info(f"Training enhanced SFA with {len(training_data)} samples")
        
        # First stage: Train base model
        X_base, y_all, tokens_all = self._prepare_base_features(training_data)
        
        # Train base meta-model
        self.meta_model = GradientBoostingRegressor(
            n_estimators=self.config.sfa_n_estimators,
            max_depth=self.config.sfa_max_depth,
            random_state=42
        )
        self.meta_model.fit(X_base, y_all)
        logger.info(f"Base model trained with score: {self.meta_model.score(X_base, y_all):.4f}")
        
        # Generate OOF predictions
        oof_predictions = self._generate_oof_predictions(X_base, y_all)
        
        # Second stage: Train augmented model with OOF predictions + Shapley values
        X_augmented = []
        for i, (prompt, shapley_dict) in enumerate(training_data):
            tokens = prompt.split()
            aug_features = self.extract_augmented_features(tokens, prompt, shapley_dict)
            
            # Add OOF predictions as features
            start_idx = sum(len(td[0].split()) for td in training_data[:i])
            end_idx = start_idx + len(tokens)
            token_predictions = oof_predictions[start_idx:end_idx]
            
            for j, feat in enumerate(aug_features):
                if j < len(token_predictions):
                    # Augment with prediction
                    aug_feat = np.append(feat, token_predictions[j])
                    X_augmented.append(aug_feat)
        
        X_augmented = np.array(X_augmented)
        
        # Train augmented model
        self.augmentation_model = RandomForestRegressor(
            n_estimators=self.config.sfa_n_estimators * 2,
            max_depth=self.config.sfa_max_depth,
            random_state=42
        )
        self.augmentation_model.fit(X_augmented, y_all)
        
        augmented_score = self.augmentation_model.score(X_augmented, y_all)
        logger.info(f"Augmented model trained with score: {augmented_score:.4f}")
        
        self.is_trained = True
        
        result = {
            'base_model_score': self.meta_model.score(X_base, y_all),
            'augmented_model_score': augmented_score,
            'n_samples': len(y_all)
        }
        
        self.training_history.append(result)
        return result
    
    def _prepare_base_features(self, training_data: List[Tuple[str, Dict[str, float]]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare base features for training"""
        all_features = []
        all_targets = []
        all_tokens = []
        
        for prompt, shapley_dict in training_data:
            tokens = prompt.split()
            features = self.extract_features(tokens, prompt)
            
            for i, token in enumerate(tokens):
                all_features.append(features[i])
                all_targets.append(shapley_dict.get(token, 0.0))
                all_tokens.append(token)
        
        return np.array(all_features), np.array(all_targets), all_tokens
    
    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold predictions for augmentation"""
        kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx] = fold_model.predict(X_val)
            
            logger.debug(f"Fold {fold_idx + 1} completed")
        
        return oof_preds
    
    def predict(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """Standard predict method for backward compatibility"""
        return self.predict_augmented(prompt, tokens)
    
    def predict_augmented(self, prompt: str, tokens: List[str], 
                         initial_shapley: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Predict using augmented features"""
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # If we have cached Shapley values, use them for augmentation
        if initial_shapley is None and prompt in self.shapley_cache:
            initial_shapley = self.shapley_cache[prompt]
        
        if self.augmentation_model and initial_shapley:
            # Use augmented model
            aug_features = self.extract_augmented_features(tokens, prompt, initial_shapley)
            
            # Add mock OOF predictions (use base model predictions)
            base_features = self.extract_features(tokens, prompt)
            base_predictions = self.meta_model.predict(base_features)
            
            full_features = []
            for i, feat in enumerate(aug_features):
                full_feat = np.append(feat, base_predictions[i])
                full_features.append(full_feat)
            
            predictions = self.augmentation_model.predict(np.array(full_features))
        else:
            # Fall back to base model
            features = self.extract_features(tokens, prompt)
            predictions = self.meta_model.predict(features)
        
        result = {token: float(pred) for token, pred in zip(tokens, predictions)}
        
        # Cache for future augmentation
        self.shapley_cache[prompt] = result
        
        return result
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.training_history:
            return {}
        
        latest = self.training_history[-1]
        return {
            'is_trained': self.is_trained,
            'training_samples': latest.get('n_samples', 0),
            'base_model_score': latest.get('base_model_score', 0.0),
            'augmented_model_score': latest.get('augmented_model_score', 0.0),
            'improvement': latest.get('augmented_model_score', 0.0) - latest.get('base_model_score', 0.0),
            'cached_predictions': len(self.shapley_cache),
            'training_iterations': len(self.training_history)
        }
    
    def save_training_data(self, filepath: str):
        """Save training data and models"""
        state = {
            'meta_model': self.meta_model,
            'augmentation_model': self.augmentation_model,
            'feature_vectorizer': self.feature_vectorizer,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'shapley_cache': self.shapley_cache,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"SFA model saved to {filepath}")
    
    def load_training_data(self, filepath: str) -> bool:
        """Load training data and models"""
        if not os.path.exists(filepath):
            logger.warning(f"SFA model file not found: {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.meta_model = state['meta_model']
            self.augmentation_model = state.get('augmentation_model')
            self.feature_vectorizer = state.get('feature_vectorizer', self.feature_vectorizer)
            self.is_trained = state['is_trained']
            self.feature_names = state['feature_names']
            self.training_history = state['training_history']
            self.shapley_cache = state.get('shapley_cache', {})
            
            logger.info(f"SFA model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SFA model: {e}")
            return False