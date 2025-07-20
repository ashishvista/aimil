#!/usr/bin/env python3
"""
Create a dummy PyTorch model to satisfy SageMaker container requirements
"""

import torch
import torch.nn as nn
import os

class DummyOCRModel(nn.Module):
    """
    A minimal dummy model that just returns a placeholder result
    """
    def __init__(self):
        super(DummyOCRModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def create_dummy_model():
    """Create and save a dummy PyTorch model"""
    
    # Create model
    model = DummyOCRModel()
    
    # Create the temp directory if it doesn't exist
    os.makedirs('temp_model', exist_ok=True)
    
    # Save the model
    model_path = 'temp_model/model.pth'
    torch.save(model.state_dict(), model_path)
    
    print(f"✅ Dummy PyTorch model created at: {model_path}")
    print(f"📊 Model file size: {os.path.getsize(model_path)} bytes")
    
    return model_path

if __name__ == "__main__":
    create_dummy_model()
