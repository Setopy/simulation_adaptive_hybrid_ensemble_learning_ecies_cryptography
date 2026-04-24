# models/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.metrics_history = []
        self.best_metrics = None
        self.best_epoch = 0
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# models/cnn_model.py
import torch
import torch.nn as nn
from .base_model import BaseModel

class CNN_IDS(BaseModel):
    def __init__(self, input_size: int):
        super(CNN_IDS, self).__init__()
        
        self.sequence_length = 14
        self.n_channels = input_size // self.sequence_length
        if input_size % self.sequence_length != 0:
            self.n_channels += 1
        
        # Feature Extraction Layers
        self.feature_extractor = nn.Sequential(
            # First Conv Block
            nn.Conv1d(self.n_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Second Conv Block with residual connection
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            # Third Conv Block with attention
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4)
        )
        
        # Self-Attention Layer
        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_channels, -1)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Apply self-attention
        attn_out, _ = self.attention(
            features.permute(0, 2, 1), 
            features.permute(0, 2, 1), 
            features.permute(0, 2, 1)
        )
        attn_out = attn_out.permute(0, 2, 1)
        
        # Global average pooling
        x = self.gap(attn_out)
        x = x.view(batch_size, -1)
        
        # Classification
        return self.classifier(x)

# models/lstm_model.py
class LSTM_IDS(BaseModel):
    def __init__(self, input_size: int):
        super(LSTM_IDS, self).__init__()
        
        self.hidden_size = 256
        self.num_layers = 2
        self.bidirectional = True
        self.dropout_rate = 0.3
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature combination layer
        self.feature_combiner = nn.Sequential(
            nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.BatchNorm1d(128)
        )
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Combine features
        combined = self.feature_combiner(context)
        
        # Classification
        return self.classifier(combined)

# models/dnn_model.py
class DNN_IDS(BaseModel):
    def __init__(self, input_size: int):
        super(DNN_IDS, self).__init__()
        
        # Create a deeper network with residual connections
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3)
            )
        ])
        
        # Output layer
        self.output_layer = nn.Linear(64, 1)
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.input_layer(x)
        
        # Process through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output
        return self.output_layer(x)
