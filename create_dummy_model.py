#!/usr/bin/env python3
"""
Create a dummy PyTorch model for SageMaker inference
"""

import torch
import torch.nn as nn
import json

class SimpleOCRModel(nn.Module):
    """Simple OCR model for demonstration"""
    
    def __init__(self, vocab_size=1000, hidden_dim=512):
        super(SimpleOCRModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Simple CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Simple text decoder
        self.decoder = nn.Sequential(
            nn.Linear(256 * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.decoder(features)
        return output

if __name__ == "__main__":
    # Create model
    model = SimpleOCRModel(vocab_size=95, hidden_dim=512)
    
    # Initialize with random weights (for demo)
    model.eval()
    
    # Save model
    torch.save(model.state_dict(), 'temp_model/model.pth')
    print("✅ Created dummy model.pth")
