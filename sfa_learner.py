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
    """Enhanced SFA Meta-Learner with 3-model augmentation approach (Claude Opus 4.1)"""
    
    def __init__(self, config: TokenSHAPConfig):
        self.config = config
        
        # Three separate models for each augmentation type (Claude Opus 4.1 approach)
        self.meta_model_p = None          # Predictions only
        self.meta_model_shap = None       # SHAP values only  
        self.meta_model_p_shap = None     # Both P and SHAP
        self.base_model = None            # Base model for OOF predictions
        
        # Feature processing
        self.feature_vectorizer = TfidfVectorizer(max_features=100)
        self.is_trained = False
        self.feature_names = []
        self.training_history = []
        self.shapley_cache = {}
        self.augmented_features_cache = {}
        self.oof_predictions_cache = {}  # Store OOF predictions for augmentation
        
        logger.info(f"Initialized enhanced SFA with 3-model approach ({config.sfa_n_estimators} estimators)")
    
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
    
    def train(self, training_data: List[Tuple[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Train using 3-model approach (Claude Opus 4.1)"""
        return self.train_with_three_augmentations(training_data)
    

    def train_with_three_augmentations(self, training_data: List[Tuple[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Train three models with P, SHAP, and P+SHAP augmentations (Claude Opus 4.1)"""
        
        logger.info(f"Training 3-model SFA approach with {len(training_data)} samples")
        
        # Stage 1: Train base model and get OOF predictions
        X_base, y_all, tokens_all = self._prepare_base_features(training_data)
        
        # Train base model with k-fold (gets OOF predictions)  
        oof_predictions = self._train_first_stage_with_oof(X_base, y_all)
        
        # Extract SHAP features from training data
        shap_features = self._extract_shap_features(training_data)
        
        # Create three augmented feature sets
        X_aug_p = np.hstack([X_base, oof_predictions.reshape(-1, 1)])         # P only
        X_aug_shap = np.hstack([X_base, shap_features])                       # SHAP only  
        X_aug_p_shap = np.hstack([X_base, oof_predictions.reshape(-1, 1), shap_features])  # P+SHAP
        
        # Train three separate models
        self.meta_model_p = GradientBoostingRegressor(
            n_estimators=self.config.sfa_n_estimators,
            max_depth=self.config.sfa_max_depth,
            learning_rate=self.config.sfa_learning_rate,
            subsample=self.config.sfa_subsample,
            random_state=self.config.sfa_random_state
        )
        self.meta_model_p.fit(X_aug_p, y_all)
        
        self.meta_model_shap = RandomForestRegressor(
            n_estimators=self.config.sfa_n_estimators,
            max_depth=self.config.sfa_max_depth,
            random_state=self.config.sfa_random_state
        )
        self.meta_model_shap.fit(X_aug_shap, y_all)
        
        self.meta_model_p_shap = GradientBoostingRegressor(
            n_estimators=self.config.sfa_n_estimators * 2,  # More estimators for complex features
            max_depth=self.config.sfa_max_depth,
            learning_rate=self.config.sfa_learning_rate,
            subsample=self.config.sfa_subsample,
            random_state=self.config.sfa_random_state
        )
        self.meta_model_p_shap.fit(X_aug_p_shap, y_all)
        
        # Store base model for predictions
        self.base_model = GradientBoostingRegressor(
            n_estimators=self.config.sfa_n_estimators,
            max_depth=self.config.sfa_max_depth,
            learning_rate=self.config.sfa_learning_rate,
            subsample=self.config.sfa_subsample,
            random_state=self.config.sfa_random_state
        )
        self.base_model.fit(X_base, y_all)
        
        self.is_trained = True
        
        # Return scores for all three models
        scores = {
            'p_score': self.meta_model_p.score(X_aug_p, y_all),
            'shap_score': self.meta_model_shap.score(X_aug_shap, y_all), 
            'p_shap_score': self.meta_model_p_shap.score(X_aug_p_shap, y_all),
            'n_samples': len(y_all)
        }
        
        self.training_history.append(scores)
        logger.info(f"3-model training completed - P: {scores['p_score']:.4f}, SHAP: {scores['shap_score']:.4f}, P+SHAP: {scores['p_shap_score']:.4f}")
        
        return scores

    def _train_first_stage_with_oof(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train first stage base model and return OOF predictions"""
        kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_model = GradientBoostingRegressor(
                n_estimators=self.config.sfa_n_estimators,
                max_depth=self.config.sfa_max_depth,
                random_state=42 + fold_idx
            )
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx] = fold_model.predict(X_val)
        
        return oof_preds

    def _extract_shap_features(self, training_data: List[Tuple[str, Dict[str, float]]]) -> np.ndarray:
        """Extract SHAP features using proper SHAP library when available"""
        try:
            import shap
            logger.info("SHAP library available - using as feature source")
            # Build base features for all tokens in the training data
            X_base, y_all, _ = self._prepare_base_features(training_data)

            # Train a small surrogate model to map base features -> token-level target
            # We use a tree-based model so TreeExplainer can provide efficient SHAP values
            surrogate_model = GradientBoostingRegressor(
                n_estimators=self.config.sfa_n_estimators,
                max_depth=self.config.sfa_max_depth,
                random_state=self.config.sfa_random_state
            )
            surrogate_model.fit(X_base, y_all)

            # Use TreeExplainer for tree models; fall back to generic Explainer if needed
            shap_values = None
            expected_value = None
            try:
                explainer = shap.TreeExplainer(surrogate_model)
                shap_values = explainer.shap_values(X_base)
                expected_value = explainer.expected_value
                logger.info("Computed SHAP values with TreeExplainer")
            except Exception as e:
                logger.warning(f"TreeExplainer failed ({e}), falling back to generic Explainer")
                explainer = shap.Explainer(surrogate_model, X_base)
                explanation = explainer(X_base)
                # Newer SHAP returns Explanation objects
                shap_values = getattr(explanation, 'values', explanation)
                expected_value = getattr(explanation, 'base_values', 0.0)

            # Normalize shapes across SHAP versions
            # - shap_values: (n_samples, n_features)
            # - expected_value/base_values: scalar or (n_samples,)
            import numpy as _np
            shap_values_arr = _np.array(shap_values)
            if shap_values_arr.ndim == 3:
                # Some SHAP versions return [class, samples, features]; use first class for regression-like case
                shap_values_arr = shap_values_arr[0]

            # Derive a single signed SHAP scalar per sample using additivity:
            #   model_output - base_value = sum(feature_shap_values)
            shap_signed_sum = shap_values_arr.sum(axis=1)

            # expected_value can be scalar, 1-element array, or per-sample
            if isinstance(expected_value, (list, tuple, _np.ndarray)):
                try:
                    base_vals = _np.array(expected_value).reshape(-1)
                    base_val = base_vals[0] if base_vals.size > 1 else float(base_vals[0])
                except Exception:
                    base_val = float(_np.mean(expected_value))
            else:
                base_val = float(expected_value) if expected_value is not None else 0.0

            # Use the signed sum as the core SHAP-derived scalar; it correlates with f(x)-E[f(x)]
            s = shap_signed_sum.astype(_np.float32)
            shap_features = _np.stack([
                s,                         # Raw signed SHAP sum per sample
                _np.abs(s),                 # Absolute magnitude
                _np.square(s),              # Squared magnitude
                _np.sign(s),                # Sign indicator
                _np.tanh(s)                 # Bounded transform
            ], axis=1)

            return shap_features.astype(_np.float32)

        except ImportError:
            logger.warning("SHAP library not available, using TokenSHAP values as approximation")
        
        # Extract features from TokenSHAP values (valid SHAP approximation)
        shap_features = []
        
        for prompt, shapley_dict in training_data:
            tokens = prompt.split()
            for token in tokens:
                shap_val = shapley_dict.get(token, 0.0)
                # Create multiple SHAP-derived features following SHAP methodology
                shap_features.append([
                    shap_val,                    # Raw SHAP value
                    abs(shap_val),               # Absolute importance
                    shap_val ** 2,               # Squared importance
                    np.sign(shap_val),           # Sign of contribution
                    np.tanh(shap_val)            # Bounded transformation
                ])
        
        return np.array(shap_features, dtype=np.float32)

    def predict_ensemble(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """Predict using ensemble of all three models (Claude Opus 4.1)"""
        
        if not (self.meta_model_p and self.meta_model_shap and self.meta_model_p_shap):
            # Fallback to standard prediction if 3-model not available
            return self.predict_augmented(prompt, tokens)
        
        # Get base features
        base_features = self.extract_features(tokens, prompt)
        
        # Get predictions from base model (for P augmentation)
        if self.base_model:
            base_preds = self.base_model.predict(base_features)
        else:
            # Use a simple heuristic if no base model
            base_preds = np.zeros(len(tokens))
        
        # Get SHAP features (use cached if available, otherwise use simple features)
        if prompt in self.shapley_cache:
            cached_shapley = self.shapley_cache[prompt]
            shap_features = np.array([[
                cached_shapley.get(token, 0.0),
                abs(cached_shapley.get(token, 0.0)),
                cached_shapley.get(token, 0.0) ** 2,
                np.sign(cached_shapley.get(token, 0.0)),
                np.tanh(cached_shapley.get(token, 0.0))
            ] for token in tokens])
        else:
            # Create simple SHAP features based on token characteristics
            shap_features = np.array([[
                0.1 * (i + 1) / len(tokens),  # Position-based
                0.1,                          # Default importance
                0.01,                         # Default squared
                1.0,                          # Default positive
                0.1                           # Default bounded
            ] for i, token in enumerate(tokens)])
        
        # Create three augmented feature sets
        X_p = np.hstack([base_features, base_preds.reshape(-1, 1)])
        X_shap = np.hstack([base_features, shap_features])
        X_p_shap = np.hstack([base_features, base_preds.reshape(-1, 1), shap_features])
        
        # Get predictions from all three models
        preds_p = self.meta_model_p.predict(X_p)
        preds_shap = self.meta_model_shap.predict(X_shap)
        preds_p_shap = self.meta_model_p_shap.predict(X_p_shap)
        
        # Average the three predictions (ensemble approach)
        ensemble_preds = np.mean([preds_p, preds_shap, preds_p_shap], axis=0)
        
        result = {token: float(pred) for token, pred in zip(tokens, ensemble_preds)}
        
        # Cache for future use
        self.shapley_cache[prompt] = result
        
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
    
    def predict_augmented(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """Predict using augmented features - now delegates to ensemble method"""
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Delegate to ensemble method for 3-model approach
        return self.predict_ensemble(prompt, tokens)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics (supports 3-model and legacy formats)"""
        if not self.training_history:
            return {}
        
        latest = self.training_history[-1]
        stats = {
            'is_trained': self.is_trained,
            'training_samples': latest.get('n_samples', 0),
            'cached_predictions': len(self.shapley_cache),
            'training_iterations': len(self.training_history)
        }
        
        # Check if this is 3-model format
        if 'p_score' in latest:
            # 3-model statistics
            stats.update({
                'p_score': latest.get('p_score', 0.0),
                'shap_score': latest.get('shap_score', 0.0),
                'p_shap_score': latest.get('p_shap_score', 0.0),
                'model_type': '3_model_sfa'
            })
            # Include legacy compatibility
            stats.update({
                'base_model_score': latest.get('base_model_score', latest.get('p_score', 0.0)),
                'augmented_model_score': latest.get('p_shap_score', 0.0),
                'improvement': latest.get('p_shap_score', 0.0) - latest.get('p_score', 0.0)
            })
        else:
            # Legacy format
            stats.update({
                'base_model_score': latest.get('base_model_score', 0.0),
                'augmented_model_score': latest.get('augmented_model_score', 0.0),
                'improvement': latest.get('augmented_model_score', 0.0) - latest.get('base_model_score', 0.0),
                'model_type': 'standard_sfa'
            })
        
        return stats
    
    def save_training_data(self, filepath: str):
        """Save training data and models (3-model format)"""
        state = {
            # 3-model architecture (Claude Opus 4.1)
            'meta_model_p': self.meta_model_p,
            'meta_model_shap': self.meta_model_shap,
            'meta_model_p_shap': self.meta_model_p_shap,
            'base_model': self.base_model,
            
            
            # Standard components
            'feature_vectorizer': self.feature_vectorizer,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'shapley_cache': self.shapley_cache,
            'oof_predictions_cache': self.oof_predictions_cache,
            'config': self.config,
            'model_type': '3_model_sfa'  # Flag for 3-model format
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"SFA model saved to {filepath}")
    
    def load_training_data(self, filepath: str) -> bool:
        """Load training data and models (supports both 3-model and legacy formats)"""
        if not os.path.exists(filepath):
            logger.warning(f"SFA model file not found: {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Check if this is 3-model format
            if state.get('model_type') == '3_model_sfa':
                logger.info("Loading 3-model SFA format")
                # Load 3-model architecture
                self.meta_model_p = state.get('meta_model_p')
                self.meta_model_shap = state.get('meta_model_shap')
                self.meta_model_p_shap = state.get('meta_model_p_shap')
                self.base_model = state.get('base_model')
                self.oof_predictions_cache = state.get('oof_predictions_cache', {})
                
            else:
                logger.info("Loading legacy SFA format")
                # Load legacy format - upgrade to 3-model if needed
                legacy_model = state.get('meta_model')
                if legacy_model:
                    self.meta_model_p_shap = legacy_model
                    logger.warning("Upgraded legacy SFA to 3-model format")
                
            # Load common components
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
