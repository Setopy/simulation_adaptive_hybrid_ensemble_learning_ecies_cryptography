import os
import logging
import torch
import joblib
import numpy as np
import time
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.optimize import minimize
from collections import deque, defaultdict
from sim_config import CONFIG
from models.cnn_model import CNN_IDS
from models.lstm_model import LSTM_IDS
from models.dnn_model import DNN_IDS

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ids_monitor.log'),
        logging.StreamHandler()
    ]
)

class EnhancedAdaptiveEnsemble:
    """Adaptive ensemble model with automatic weight optimization and enhanced metrics tracking"""
    
    def __init__(self, config, window_size=1000):
        """
        Initialize the adaptive ensemble
        
        Args:
            config: Configuration dictionary
            window_size: Size of the sliding window for adaptation
        """
        self.base_weights = config['MODEL']['WEIGHTS']
        self.models = {}
        self.current_weights = self.base_weights.copy()
        self.meta_model = None
        self.is_meta_model_fitted = False
        
        # Performance tracking
        self.window_size = window_size
        self.predictions_history = {model: deque(maxlen=window_size) 
                                   for model in self.base_weights}
        self.true_labels = deque(maxlen=window_size)
        self.optimization_threshold = 100  # Min samples before optimization
        
        # Track metrics
        self.metrics_history = {
            'f1': [],
            'precision': [],
            'recall': [],
            'weights': []
        }
        
        # Metrics tracking enhancements
        self.model_predictions = defaultdict(list)
        self.ensemble_probabilities = []
        self.feature_importances = {}
        
        logging.info("Adaptive ensemble initialized with base weights")
        logging.info(f"Will start adapting after {self.optimization_threshold} labeled samples")
        
    def register_models(self, models):
        """Register models with the ensemble"""
        self.models = models
        logging.info(f"Registered {len(models)} models with the ensemble")
    
    def predict(self, features, true_label=None):
        """
        Make a prediction using the current weights or meta-model
        
        Args:
            features: Input features
            true_label: If provided, used for adaptation
            
        Returns:
            Ensemble prediction and probability
        """
        if not self.models:
            logging.error("No models registered with ensemble")
            return False, 0.0
            
        predictions = {}
        features_tensor = torch.FloatTensor(features)
        features_array = features.reshape(1, -1)
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                # Different prediction method based on model type
                if isinstance(model, torch.nn.Module):
                    with torch.no_grad():
                        outputs = model(features_tensor)
                        prediction_value = torch.sigmoid(outputs).item()
                
                # SVM with decision function
                elif hasattr(model, 'decision_function'):
                    decision = model.decision_function(features_array)[0]
                    # Safe sigmoid implementation
                    prediction_value = 1 / (1 + np.exp(-np.clip(decision, -709, 709)))
                
                # XGBoost and Random Forest with predict_proba
                elif hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_array)
                    prediction_value = proba[0][1]  # Probability of intrusion class
                
                else:
                    logging.warning(f"Unknown model type for {name}, using default prediction")
                    prediction_value = 0.5
                
                # Set prediction value    
                predictions[name] = prediction_value
                
                # Store prediction in model object for easy access
                model.last_prediction = prediction_value
                
                # Store prediction history
                self.predictions_history[name].append(prediction_value)
                
                # Enhanced metrics tracking
                self.model_predictions[name].append(prediction_value)
                
                # Log the prediction
                logging.info(f"{name.upper()} prediction: {prediction_value:.6f}")
                
            except Exception as e:
                logging.error(f"Error getting prediction from {name}: {str(e)}")
                predictions[name] = 0.5
                self.predictions_history[name].append(0.5)
                model.last_prediction = 0.5
        
        # If we have a trained meta-model, use it
        if self.is_meta_model_fitted:
            try:
                # Prepare input for meta-model (ensure columns are in the right order)
                meta_features = []
                for name in sorted(self.current_weights.keys()):
                    if name in predictions:
                        meta_features.append(predictions[name])
                    else:
                        meta_features.append(0.5)  # Default if model is missing
                
                meta_input = np.array([meta_features])
                ensemble_prob = self.meta_model.predict_proba(meta_input)[0][1]
                
            except Exception as e:
                logging.error(f"Error using meta-model: {str(e)}. Falling back to weighted average.")
                # Fallback to weighted average
                ensemble_prob = sum(predictions.get(name, 0) * self.current_weights.get(name, 0) 
                                  for name in predictions)
        else:
            # Use weighted average
            ensemble_prob = sum(predictions.get(name, 0) * self.current_weights.get(name, 0) 
                              for name in predictions)
        
        # Log model contributions
        for name in self.current_weights:
            weight = self.current_weights.get(name, 0)
            logging.info(f"{name.upper()} contribution: {weight:.6f}")
        
        # Log the ensemble probability
        logging.info(f"Final ensemble probability: {ensemble_prob:.6f}")
        
        # Track ensemble probability
        self.ensemble_probabilities.append(ensemble_prob)
        
        # If true label is provided, store it for adaptation
        if true_label is not None:
            self.true_labels.append(int(true_label))  # Ensure label is an integer
            
            # Log progress toward adaptation
            if not self.is_meta_model_fitted and len(self.true_labels) % 10 == 0:
                logging.info(f"Collected {len(self.true_labels)}/{self.optimization_threshold} labeled samples for adaptation")
            
            # Check if we have enough history to optimize
            if len(self.true_labels) >= self.optimization_threshold:
                self._adapt_weights()
        
        threshold = CONFIG['THRESHOLDS']['INTRUSION_PROBABILITY']
        return ensemble_prob > threshold, ensemble_prob
    
    def _adapt_weights(self):
        """Adapt ensemble weights based on recent performance"""
        try:
            # Only adapt if we have enough history
            if len(self.true_labels) < self.optimization_threshold:
                return
                
            # Log that we're attempting to adapt weights
            logging.info(f"Attempting to adapt weights with {len(self.true_labels)} samples")
                
            # Convert deques to arrays for processing
            y_true = np.array(list(self.true_labels))
            
            # Check for data issues
            if len(np.unique(y_true)) < 2:
                logging.warning(f"Cannot adapt weights: Need both positive and negative samples. Current labels: {np.unique(y_true)}")
                return
            
            # Prepare model predictions for meta-model
            X_meta = []
            model_names = sorted(self.models.keys())
            
            for i in range(len(y_true)):
                sample_preds = []
                for name in model_names:
                    # Ensure we have enough predictions for this model
                    if i < len(self.predictions_history[name]):
                        sample_preds.append(self.predictions_history[name][i])
                    else:
                        sample_preds.append(0.5)  # Default
                X_meta.append(sample_preds)
            
            X_meta = np.array(X_meta)
            
            # Validate dimensions
            if X_meta.shape[0] != y_true.shape[0]:
                logging.error(f"Data dimension mismatch: X_meta shape {X_meta.shape}, y_true shape {y_true.shape}")
                return
                
            # Train meta-learner (stacking)
            self.meta_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
            self.meta_model.fit(X_meta, y_true)
            self.is_meta_model_fitted = True
            
            # Update weights based on meta-model coefficients
            # Get abs coefficients and normalize to sum to 1
            coefs = np.abs(self.meta_model.coef_[0])
            sum_coefs = np.sum(coefs)
            
            if sum_coefs > 0:  # Avoid division by zero
                normalized_coefs = coefs / sum_coefs
                self.current_weights = {name: coef for name, coef in 
                                      zip(model_names, normalized_coefs)}
                
                # Calculate and store metrics
                meta_preds = self.meta_model.predict(X_meta)
                f1 = f1_score(y_true, meta_preds)
                precision = precision_score(y_true, meta_preds)
                recall = recall_score(y_true, meta_preds)
                
                self.metrics_history['f1'].append(f1)
                self.metrics_history['precision'].append(precision)
                self.metrics_history['recall'].append(recall)
                self.metrics_history['weights'].append(self.current_weights.copy())
                
                logging.info(f"Successfully adapted ensemble weights: {self.current_weights}")
                logging.info(f"New ensemble metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            else:
                logging.warning("All coefficients are zero. Keeping original weights.")
                
        except Exception as e:
            logging.error(f"Error during weight adaptation: {str(e)}")
            logging.error("Traceback:", exc_info=True)  # Print full traceback
            # Fallback to base weights
            self.current_weights = self.base_weights.copy()
            self.is_meta_model_fitted = False
    
    def get_current_weights(self):
        """Get current ensemble weights"""
        return self.current_weights
    
    def get_metrics_history(self):
        """Get metrics history"""
        return self.metrics_history
    
    def get_model_predictions(self):
        """Get the history of model predictions"""
        return {name: list(values) for name, values in self.model_predictions.items()}
    
    def get_ensemble_probabilities(self):
        """Get the history of ensemble probabilities"""
        return list(self.ensemble_probabilities)
    
    def log_feature_importances(self, feature_names):
        """Log feature importances for models that support it"""
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Create dict of feature name -> importance
                feature_dict = {}
                for i, importance in enumerate(importances):
                    feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    feature_dict[feature_name] = float(importance)
                
                # Sort by importance (descending)
                sorted_features = sorted(
                    feature_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Store for later access
                self.feature_importances[name] = sorted_features
                
                # Log top 5 features
                logging.info(f"{name.upper()} top feature importances:")
                for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                    logging.info(f"  {i}. {feature}: {importance:.6f}")
            
            elif hasattr(model, 'coef_'):
                # For linear models (SVM, etc.)
                coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                # Use absolute values for importance
                importances = np.abs(coefs)
                
                # Create dict of feature name -> importance
                feature_dict = {}
                for i, importance in enumerate(importances):
                    feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    feature_dict[feature_name] = float(importance)
                
                # Sort by importance (descending)
                sorted_features = sorted(
                    feature_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Store for later access
                self.feature_importances[name] = sorted_features
                
                # Log top 5 features
                logging.info(f"{name.upper()} top feature importances:")
                for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                    logging.info(f"  {i}. {feature}: {importance:.6f}")
    
    def optimize_weights_bayesian(self, X, y):
        """Use Bayesian optimization for weight optimization"""
        # This would be a more advanced implementation using libraries like scikit-optimize
        # For simplicity, we'll use scipy's minimize for demonstration
        
        def objective(weights):
            # Normalize weights to sum to 1
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            
            # Make predictions
            y_probs = np.zeros(len(y))
            for i, model_name in enumerate(sorted(self.models.keys())):
                model_preds = np.array(self.predictions_history[model_name])
                y_probs += weights[i] * model_preds
                
            # Convert to binary predictions
            y_pred = (y_probs > 0.5).astype(int)
            
            # Return negative F1 score (minimize negative = maximize positive)
            return -f1_score(y, y_pred)
        
        # Initial weights
        initial_weights = np.array([self.current_weights[name] for name in sorted(self.models.keys())])
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=[(0.01, 1.0) for _ in range(len(initial_weights))],
                constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            )
            
            if result.success:
                # Update weights
                optimized_weights = np.abs(result.x)
                optimized_weights = optimized_weights / np.sum(optimized_weights)
                self.current_weights = {name: weight for name, weight in 
                                     zip(sorted(self.models.keys()), optimized_weights)}
                logging.info(f"Bayesian optimized weights: {self.current_weights}")
                return True
            else:
                logging.warning(f"Weight optimization failed: {result.message}")
                return False
                
        except Exception as e:
            logging.error(f"Error in Bayesian optimization: {str(e)}")
            return False

class EnhancedIDSMonitor:
    """Enhanced Intrusion Detection System Monitor with comprehensive metrics tracking"""
    
    def __init__(self, models_dir: str):
        """
        Initialize IDS with pre-trained models
        
        Args:
            models_dir (str): Directory containing the model files
        """
        logging.info("Initializing IDSMonitor with Adaptive Ensemble")
        self.models_dir = models_dir
        self.models = {}
        self.model_type = None
        self.input_size = 196  # Set based on your feature count
        
        # Verify models directory exists
        if not os.path.exists(models_dir):
            logging.error(f"Models directory not found: {models_dir}")
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Log available model files
        model_files = os.listdir(models_dir)
        logging.debug(f"Available model files: {model_files}")
        
        # Load models
        self.models = self._load_model()
        
        # Create the adaptive ensemble (using enhanced version)
        self.ensemble = EnhancedAdaptiveEnsemble(CONFIG)
        self.ensemble.register_models(self.models)
        
        # Track recent detection history
        self.true_positive_history = []
        self.false_positive_history = []
        self.threshold_history = []
        self.prediction_confidence = []
        
        # Store threshold for easy access
        self.threshold = CONFIG['THRESHOLDS']['INTRUSION_PROBABILITY']
        
        # Track last probability
        self.last_probability = 0.0

    def _load_model(self):
        """Load the trained models"""
        try:
            models = {}
        
            # Load neural models
            for name in ['cnn', 'lstm', 'dnn']:
                try:
                    model_path = os.path.join(self.models_dir, f"{name}_model.pth")
                    if os.path.exists(model_path):
                        logging.info(f"Loading {name} model from {model_path}")
                        # Load the checkpoint
                        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                        
                        # Debug print for checkpoint structure
                        logging.debug(f"Checkpoint keys for {name}: {checkpoint.keys()}")
                        
                        # Extract the actual state dict
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                            logging.debug(f"Model state dict keys for {name}: {state_dict.keys()}")
                            
                            # Create model instance based on type
                            if name == 'cnn':
                                model = CNN_IDS(self.input_size)
                            elif name == 'lstm':
                                model = LSTM_IDS(self.input_size)
                            else:  # dnn
                                model = DNN_IDS(self.input_size)
                            
                            # Load the state dict
                            try:
                                model.load_state_dict(state_dict)
                                model.eval()
                                models[name] = model
                                
                                # Log successful loading with model details
                                logging.info(f"Successfully loaded {name.upper()} model")
                                if 'config' in checkpoint:
                                    logging.debug(f"Model config: {checkpoint['config']}")
                            except Exception as e:
                                logging.error(f"Error loading state dict for {name} model: {str(e)}")
                        else:
                            raise ValueError(f"'model_state_dict' not found in {name} checkpoint")
                            
                except Exception as e:
                    logging.error(f"Error loading {name} model: {str(e)}")
                    logging.error("Traceback: ", exc_info=True)
        
            # Load traditional models
            for name, path in [
                ('xgboost', 'XGBoost/model.joblib'),
                ('randomforest', 'RandomForest/model.joblib'),
                ('svm', 'SVM/svm_model.joblib'),
                ('xgboost', 'XGBoost/xgboost_model.joblib'),
                ('randomforest', 'RandomForest/randomforest_model.joblib')
            ]:
                try:
                    model_path = os.path.join(self.models_dir, path)
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        models[name] = model
                        logging.info(f"Successfully loaded {name.upper()} model")
                except Exception as e:
                    logging.error(f"Error loading {name} model: {str(e)}")
        
            if len(models) > 0:
                logging.info(f"Loaded ensemble of {len(models)} models successfully")
                return models
            else:
                raise ValueError("No models were successfully loaded")
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def preprocess_traffic(self, traffic_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Preprocess traffic data for model input"""
        try:
            features = []
            feature_names = []
            
            # Process numeric features
            numeric_features = sorted([k for k in traffic_data.keys() 
                                    if not k.startswith(('proto_', 'service_', 'state_'))])
            for feat in numeric_features:
                features.append(float(traffic_data.get(feat, 0)))
                feature_names.append(feat)
            
            # Process protocol features
            proto_features = sorted([k for k in traffic_data.keys() 
                                  if k.startswith('proto_')])
            for feat in proto_features:
                features.append(float(traffic_data.get(feat, 0)))
                feature_names.append(feat)
            
            # Process service features
            service_features = sorted([k for k in traffic_data.keys() 
                                    if k.startswith('service_')])
            for feat in service_features:
                features.append(float(traffic_data.get(feat, 0)))
                feature_names.append(feat)
            
            # Process state features
            state_features = sorted([k for k in traffic_data.keys() 
                                  if k.startswith('state_')])
            for feat in state_features:
                features.append(float(traffic_data.get(feat, 0)))
                feature_names.append(feat)
            
            # Debug output
            if CONFIG['DEBUG']['PRINT_FEATURES']:
                logging.debug(f"\nPreprocessed features:")
                logging.debug(f"  Numeric features: {len(numeric_features)}")
                logging.debug(f"  Protocol features: {len(proto_features)}")
                logging.debug(f"  Service features: {len(service_features)}")
                logging.debug(f"  State features: {len(state_features)}")
                logging.debug(f"  Total features: {len(features)}")
            
            return np.array(features).reshape(1, -1), feature_names
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def detect_intrusion(self, traffic_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Detect if traffic is malicious using the adaptive ensemble.
        Returns a tuple of (is_intrusion, probability).
        """
        try:
            # Preprocess features as before
            features, feature_names = self.preprocess_traffic(traffic_data)
            
            if CONFIG['DEBUG']['PRINT_FEATURES']:
                logging.debug(f"Feature vector shape: {features.shape}")
            
            # Log feature importances on first access
            if not hasattr(self.ensemble, 'has_logged_importances'):
                self.ensemble.log_feature_importances(feature_names)
                self.ensemble.has_logged_importances = True
            
            # Use the ensemble for prediction
            is_intrusion, probability = self.ensemble.predict(features)
            
            # Save the probability for reference
            self.last_probability = probability
            
            # Track detection information for analytics
            if is_intrusion:
                # If we have ground truth, track true/false positives
                if 'label' in traffic_data:
                    if traffic_data.get('label', 0) == 1:
                        self.true_positive_history.append((time.time(), probability))
                    else:
                        self.false_positive_history.append((time.time(), probability))
            
            # Track all prediction confidences
            self.prediction_confidence.append(probability)
            
            # If we have ground truth, update the ensemble for adaptation
            if 'label' in traffic_data:
                true_label = traffic_data['label']
                # This doesn't change the current prediction but updates for next time
                self.ensemble.predict(features, true_label=true_label)
            
            # Log model decisions if debug enabled
            if CONFIG['DEBUG']['LOG_MODEL_DECISIONS']:
                current_weights = self.ensemble.get_current_weights()
                logging.debug(f"Current ensemble weights: {current_weights}")
                logging.debug(f"Final probability: {probability:.4f}")
                
            # Log if intrusion detected
            if is_intrusion:
                logging.warning(f"Intrusion confirmed! Confidence: {probability:.4f}")
            
            return is_intrusion, probability
                
        except Exception as e:
            logging.error(f"Error in intrusion detection: {str(e)}")
            return False, 0.0

    def get_feature_importance(self, features: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
        """Get feature importance for the prediction"""
        try:
            importances = {}
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    model_importances = model.feature_importances_
                    for feat, imp in zip(feature_names, model_importances):
                        importances[feat] = importances.get(feat, 0) + imp
            
            if importances:
                # Average the importances
                for feat in importances:
                    importances[feat] /= len([m for m in self.models.values() 
                                           if hasattr(m, 'feature_importances_')])
                return sorted(importances.items(), key=lambda x: x[1], reverse=True)
            return []
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return []

    def analyze_traffic_pattern(self, traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze traffic pattern for anomalies"""
        analysis = {
            'high_volume': traffic_data.get('sbytes', 0) > CONFIG['THRESHOLDS']['HIGH_VOLUME_BYTES'] or 
                         traffic_data.get('dbytes', 0) > CONFIG['THRESHOLDS']['HIGH_VOLUME_BYTES'],
            'high_error_rate': traffic_data.get('serror_rate', 0) > CONFIG['THRESHOLDS']['HIGH_ERROR_RATE'] or 
                             traffic_data.get('rerror_rate', 0) > CONFIG['THRESHOLDS']['HIGH_ERROR_RATE'],
            'suspicious_service': traffic_data.get('service_', '') in ['ftp', 'ssh'],
            'suspicious_state': traffic_data.get('state_', '') in ['RST', 'REJ']
        }
        return analysis
        
    def get_ensemble_metrics(self):
        """Get enhanced ensemble metrics for monitoring and visualization"""
        # Get base metrics
        base_metrics = {
            'weights': self.ensemble.get_current_weights(),
            'metrics_history': self.ensemble.get_metrics_history(),
            'true_positives': len(self.true_positive_history),
            'false_positives': len(self.false_positive_history),
            'average_confidence': np.mean(self.prediction_confidence) if self.prediction_confidence else 0,
            'confidence_std': np.std(self.prediction_confidence) if self.prediction_confidence else 0
        }
        
        # Add enhanced metrics
        enhanced_metrics = {
            'model_performance': {},
            'ensemble': {
                'values': self.ensemble.get_ensemble_probabilities(),
                'mean': np.mean(self.prediction_confidence) if self.prediction_confidence else 0,
                'std': np.std(self.prediction_confidence) if self.prediction_confidence else 0
            },
            'threshold': self.threshold,
            'feature_importances': {
                name: importances for name, importances in self.ensemble.feature_importances.items()
            }
        }
        
        # Add model-specific performance metrics
        model_predictions = self.ensemble.get_model_predictions()
        current_weights = self.ensemble.get_current_weights()
        
        for model_name in self.models:
            predictions = model_predictions.get(model_name, [])
            
            # Skip if no predictions
            if not predictions:
                continue
                
            # Extract prediction stats
            model_metrics = {
                'predictions': {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions)),
                    'count': len(predictions)
                },
                'contributions': {
                    'mean': float(current_weights.get(model_name, 0)),
                    'std': 0,  # Static weight
                    'min': float(current_weights.get(model_name, 0)),
                    'max': float(current_weights.get(model_name, 0)),
                    'count': 1
                }
            }
            
            # Add to metrics
            enhanced_metrics['model_performance'][model_name] = model_metrics
        
        # Combine base and enhanced metrics
        return {**base_metrics, **enhanced_metrics}