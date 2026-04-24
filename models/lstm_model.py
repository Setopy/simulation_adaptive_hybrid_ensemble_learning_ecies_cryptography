import torch
import torch.nn as nn
from .base_model import BaseModel
from config import CONFIG

class LSTM_IDS(BaseModel):
    def __init__(self, input_size: int):
        super(LSTM_IDS, self).__init__()
        
        # Hyperparameters
        self.hidden_size = 256  # Increased from 128
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
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        for m in self.feature_combiner.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
