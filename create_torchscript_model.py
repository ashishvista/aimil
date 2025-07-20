#!/usr/bin/env python3
"""
Create a TorchScript model for SageMaker OCR endpoint
This creates a proper TorchScript model file instead of just state_dict
"""

import torch
import torch.nn as nn
import os

class DummyOCRModel(nn.Module):
    """
    Simple dummy OCR model that can be converted to TorchScript
    """
    def __init__(self, vocab_size=1000, hidden_dim=128):
        super(DummyOCRModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Simple layers for demonstration
        self.feature_extractor = nn.Sequential(
            nn.Linear(784, hidden_dim),  # Assuming flattened 28x28 input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass - must be TorchScript compatible
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        probabilities = self.softmax(logits)
        
        return probabilities

def create_torchscript_model():
    """
    Create and save a TorchScript model
    """
    print("Creating DummyOCRModel...")
    
    # Create model instance
    model = DummyOCRModel(vocab_size=1000, hidden_dim=128)
    
    # Put model in evaluation mode
    model.eval()
    
    # Create example input for tracing (batch_size=1, flattened 28x28 image)
    example_input = torch.randn(1, 784)
    
    print("Converting to TorchScript...")
    
    # Option 1: Use torch.jit.trace
    try:
        traced_model = torch.jit.trace(model, example_input)
        print("✅ Successfully traced model")
        
        # Test the traced model
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            print(f"Original output shape: {original_output.shape}")
            print(f"Traced output shape: {traced_output.shape}")
            print(f"Outputs match: {torch.allclose(original_output, traced_output)}")
        
        return traced_model
        
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        print("Trying torch.jit.script instead...")
        
        # Option 2: Use torch.jit.script
        try:
            scripted_model = torch.jit.script(model)
            print("✅ Successfully scripted model")
            
            # Test the scripted model
            with torch.no_grad():
                original_output = model(example_input)
                scripted_output = scripted_model(example_input)
                print(f"Original output shape: {original_output.shape}")
                print(f"Scripted output shape: {scripted_output.shape}")
                print(f"Outputs match: {torch.allclose(original_output, scripted_output)}")
            
            return scripted_model
            
        except Exception as e:
            print(f"❌ Scripting also failed: {e}")
            raise

def main():
    # Create temp_model directory if it doesn't exist
    temp_dir = "/Users/ashish_kumar/aiml/temp_model"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create TorchScript model
    torchscript_model = create_torchscript_model()
    
    # Save the TorchScript model
    model_path = os.path.join(temp_dir, "model.pth")
    
    print(f"Saving TorchScript model to {model_path}")
    torch.jit.save(torchscript_model, model_path)
    
    # Verify the saved model can be loaded
    print("Verifying saved model...")
    loaded_model = torch.jit.load(model_path)
    
    # Test the loaded model
    test_input = torch.randn(1, 784)
    with torch.no_grad():
        output = loaded_model(test_input)
        print(f"✅ Loaded model works! Output shape: {output.shape}")
        print(f"Output sample: {output[0][:5].tolist()}")
    
    print(f"\n✅ TorchScript model successfully created and saved!")
    print(f"Model file: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
