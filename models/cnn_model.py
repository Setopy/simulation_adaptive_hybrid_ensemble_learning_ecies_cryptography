import torch
import torch.nn as nn
from .base_model import BaseModel
from config import CONFIG

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
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input for CNN
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
