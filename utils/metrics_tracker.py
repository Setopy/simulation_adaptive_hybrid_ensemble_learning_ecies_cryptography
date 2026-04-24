import os
import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from config import CONFIG

class MetricsTracker:
    """Track and save training metrics"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_history = []
        self.model_dir = os.path.join(CONFIG['PATHS']['METRICS'], model_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def _standardize_metric_names(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Ensure all required metrics exist with correct names"""
        standardized = {
            'epoch': metrics['epoch'],
            'timestamp': metrics['timestamp'],
            'accuracy': float(metrics.get('test_accuracy', 0.0)),
            'f1_score': float(metrics.get('test_f1', 0.0)),
            'train_accuracy': float(metrics.get('train_accuracy', 0.0)),
            'test_accuracy': float(metrics.get('test_accuracy', 0.0)),
            'train_f1': float(metrics.get('train_f1', 0.0)),
            'test_f1': float(metrics.get('test_f1', 0.0)),
            'train_loss': float(metrics.get('train_loss', 0.0)),
            'test_loss': float(metrics.get('test_loss', 0.0)),
            'learning_rate': float(metrics.get('learning_rate', 0.0))
        }
        return standardized

    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update metrics history"""
        metrics_with_meta = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        standardized_metrics = self._standardize_metric_names(metrics_with_meta)
        self.metrics_history.append(standardized_metrics)
        self._save_metrics()

    def _get_best_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get best metrics across all epochs"""
        return {
            'best_train_accuracy': df['train_accuracy'].max(),
            'best_test_accuracy': df['test_accuracy'].max(),
            'best_train_f1': df['train_f1'].max(),
            'best_test_f1': df['test_f1'].max(),
            'best_train_loss': df['train_loss'].min(),
            'best_test_loss': df['test_loss'].min(),
            'final_learning_rate': df['learning_rate'].iloc[-1]
        }

    def _save_metrics(self):
        """Save metrics to files"""
        df = pd.DataFrame(self.metrics_history)
        
        # Save detailed metrics to CSV
        csv_path = os.path.join(self.model_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)

        # Save summary to JSON
        summary = {
            'model_name': self.model_name,
            'last_epoch': self.metrics_history[-1]['epoch'],
            'best_metrics': self._get_best_metrics(df),
            'training_history': {
                'start_time': self.metrics_history[0]['timestamp'],
                'end_time': self.metrics_history[-1]['timestamp'],
                'total_epochs': len(self.metrics_history)
            }
        }

        json_path = os.path.join(self.model_dir, 'training_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent metrics"""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]

    def get_all_metrics(self) -> pd.DataFrame:
        """Get all historical metrics as a DataFrame"""
        return pd.DataFrame(self.metrics_history)
