import torch
import torch.nn as nn
from .base_model import BaseModel
from config import CONFIG

class DNN_IDS(BaseModel):
    """Improved Deep Neural Network model for Intrusion Detection"""
    def __init__(self, input_size: int):
        super(DNN_IDS, self).__init__()
        
        # Get parameters from config
        params = CONFIG['MODEL_PARAMS']['DNN']
        
        # Create a deeper network with better regularization
        layers = []
        current_size = input_size
        
        # First layer - reduce dimensionality
        layers.extend([
            nn.Linear(current_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        ])
        current_size = 512
        
        # Hidden layers with residual connections
        hidden_sizes = [256, 128, 64]
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(0.3)
            ])
            current_size = size
            
        # Additional layer for better feature extraction
        layers.extend([
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        ])
        
        # Output layer
        layers.extend([
            nn.Linear(32, 1)
            # Note: No sigmoid here as we're using BCEWithLogitsLoss
        ])
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
