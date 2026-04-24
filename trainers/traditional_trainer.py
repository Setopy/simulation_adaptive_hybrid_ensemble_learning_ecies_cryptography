import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics_tracker import MetricsTracker
from config import CONFIG
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from utils.metrics_tracker import MetricsTracker
import logging
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

class TraditionalTrainer:
    """Trainer class for traditional ML models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_tracker = MetricsTracker(model_name)
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics and convert to regular float"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred))
        }
        return metrics
    
    def train(self, model: Any,
              X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """Train traditional model and track metrics"""
        
        try:
            logging.info(f"Training {self.model_name}...")
            
            # Training
            if isinstance(model, XGBClassifier):
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
            
            elif isinstance(model, RandomForestClassifier):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Feature importance analysis
                if hasattr(model, 'feature_importances_'):
                    importances = pd.Series(model.feature_importances_)
                    logging.info(f"Top 10 important features:\n{importances.nlargest(10)}")
            
            elif isinstance(model, LinearSVC):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            logging.info(f"{self.model_name} training completed")
            logging.info(f"Test metrics: {metrics}")
            
            return model, metrics
            
        except Exception as e:
            logging.error(f"Error training {self.model_name}: {str(e)}")
            raise
