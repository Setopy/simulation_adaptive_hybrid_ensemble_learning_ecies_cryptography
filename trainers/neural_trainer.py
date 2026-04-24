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

class NeuralTrainer:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.metrics_tracker = MetricsTracker(model_name)
        self.early_stopping_patience = CONFIG['EARLY_STOPPING_PATIENCE']
        self.best_model_state = None
        self.best_f1 = 0
        self.patience_counter = 0
        self.scaler = GradScaler()
        
        # Add gradient clipping parameters
        self.max_grad_norm = 1.0
        self.clip_value = 5.0
    
    def _prepare_data_loader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Prepare DataLoader for training"""
        tensor_X = torch.FloatTensor(X)
        tensor_y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        dataset = TensorDataset(tensor_X, tensor_y)
        return DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics and convert to regular float"""
        try:
            # Convert inputs to flat arrays
            y_true = y_true.ravel()
            y_pred = y_pred.ravel()
            
            # Ensure binary values
            y_pred = (y_pred > 0.5).astype(int)
            y_true = y_true.astype(int)
            
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
            }
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
    def train(self, model: nn.Module,
              X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              gradient_accumulation_steps: int = 4) -> Tuple[nn.Module, Dict[str, float]]:
        try:
            logging.info(f"Training {self.model_name}...")
            
            # Prepare data
            train_loader = self._prepare_data_loader(X_train, y_train)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test.values if hasattr(y_test, 'values') else y_test).to(self.device)
            
            # Setup model and training components
            model = model.to(self.device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
            
            # Setup learning rate scheduler
            steps_per_epoch = len(train_loader) // gradient_accumulation_steps
            if steps_per_epoch == 0:
                steps_per_epoch = 1
                
            scheduler = OneCycleLR(
                optimizer,
                max_lr=CONFIG['LEARNING_RATE'],
                epochs=CONFIG['NUM_EPOCHS'],
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            
            # Training loop
            for epoch in range(CONFIG['NUM_EPOCHS']):
                model.train()
                train_loss = 0
                train_predictions = []
                train_targets = []
                optimizer.zero_grad()
                
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Mixed precision training
                    with autocast():
                        logits = model(batch_X)
                        loss = criterion(logits, batch_y.unsqueeze(1))
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.scaler.unscale_(optimizer)
                        
                        # Value-based gradient clipping
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(),
                            self.clip_value
                        )
                        
                        # Norm-based gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.max_grad_norm,
                            norm_type=2.0
                        )
                        
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    # Store predictions
                    train_loss += loss.item() * gradient_accumulation_steps
                    predictions = torch.sigmoid(logits).detach()
                    train_predictions.extend(predictions.cpu().numpy())
                    train_targets.extend(batch_y.cpu().numpy())
                
                # Calculate training metrics
                train_predictions = np.array(train_predictions).reshape(-1)
                train_targets = np.array(train_targets).reshape(-1)
                train_pred_binary = (train_predictions > 0.5).astype(int)
                train_metrics = self._calculate_metrics(train_targets, train_pred_binary)
                train_metrics['loss'] = train_loss / len(train_loader)
                
                # Validation phase
                model.eval()
                with torch.no_grad():
                    test_logits = model(X_test_tensor)
                    test_loss = criterion(test_logits, y_test_tensor.unsqueeze(1)).item()
                    test_predictions = torch.sigmoid(test_logits).cpu().numpy().reshape(-1)
                    test_pred_binary = (test_predictions > 0.5).astype(int)
                    test_metrics = self._calculate_metrics(
                        y_test.values if hasattr(y_test, 'values') else y_test,
                        test_pred_binary
                    )
                    test_metrics['loss'] = test_loss
                
                # Update metrics tracker
                epoch_metrics = {
                    'train_loss': train_metrics['loss'],
                    'test_loss': test_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'test_accuracy': test_metrics['accuracy'],
                    'train_f1': train_metrics['f1_score'],
                    'test_f1': test_metrics['f1_score'],
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                self.metrics_tracker.update(epoch, epoch_metrics)
                
                # Early stopping check
                if test_metrics['f1_score'] > self.best_f1:
                    self.best_f1 = test_metrics['f1_score']
                    self.best_model_state = model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Log progress
                if (epoch + 1) % 5 == 0:
                    logging.info(
                        f"Epoch {epoch+1}: "
                        f"Train Loss = {train_metrics['loss']:.4f}, "
                        f"Test Loss = {test_metrics['loss']:.4f}, "
                        f"Train F1 = {train_metrics['f1_score']:.4f}, "
                        f"Test F1 = {test_metrics['f1_score']:.4f}, "
                        f"LR = {scheduler.get_last_lr()[0]:.6f}"
                    )
                
                if self.patience_counter >= self.early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Load best model and perform final evaluation
            if self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
            
            return model, test_metrics
            
        except Exception as e:
            logging.error(f"Error training {self.model_name}: {str(e)}")
            raise
    
    def save_model(self, model: nn.Module, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_f1': self.best_f1,
            'config': {
                'model_name': self.model_name,
                'input_size': next(model.parameters()).size(1)
            }
        }, path)
