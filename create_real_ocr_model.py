#!/usr/bin/env python3
"""
Create a real PyTorch OCR model for demonstration
This creates a simple CRNN-style model for text recognition
"""

import torch
import torch.nn as nn
import json
import os
import tarfile
from pathlib import Path

class SimpleOCRModel(nn.Module):
    """
    A simple OCR model for demonstration
    In reality, this would be a more sophisticated CRNN or Transformer model
    """
    def __init__(self, vocab_size=128, hidden_size=256, num_layers=2):
        super(SimpleOCRModel, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # RNN sequence processor
        self.rnn = nn.LSTM(128, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Output layer
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
    def forward(self, x):
        # Extract CNN features
        batch_size = x.size(0)
        cnn_features = self.cnn(x)  # [B, 128, H', W']
        
        # Reshape for RNN: [B, W', 128*H'] 
        b, c, h, w = cnn_features.size()
        cnn_features = cnn_features.permute(0, 3, 1, 2)  # [B, W', C, H']
        cnn_features = cnn_features.contiguous().view(b, w, c * h)
        
        # Process with RNN
        rnn_output, _ = self.rnn(cnn_features)  # [B, W', hidden_size*2]
        
        # Classify each position
        output = self.classifier(rnn_output)  # [B, W', vocab_size]
        
        return output

def create_vocabulary():
    """Create a character vocabulary for OCR"""
    # Basic ASCII characters commonly found in text
    chars = []
    
    # Add special tokens
    chars.extend(['<blank>', '<unk>', '<eos>'])
    
    # Add digits
    chars.extend([str(i) for i in range(10)])
    
    # Add uppercase letters
    chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    
    # Add lowercase letters
    chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
    
    # Add common punctuation and symbols
    chars.extend([' ', '.', ',', '!', '?', ':', ';', '-', '_', '/', '\\', 
                  '(', ')', '[', ']', '{', '}', '"', "'", '@', '#', '$', 
                  '%', '&', '*', '+', '=', '<', '>', '|', '~', '`'])
    
    # Create char to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return {
        'chars': chars,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': len(chars)
    }

def create_model_config():
    """Create model configuration"""
    return {
        'model_type': 'simple_ocr_crnn',
        'version': '1.0.0',
        'input_size': [3, 64, 256],  # [C, H, W] - typical OCR input size
        'hidden_size': 256,
        'num_layers': 2,
        'description': 'Simple CRNN-style model for OCR text recognition',
        'training_info': {
            'framework': 'pytorch',
            'created_date': '2025-01-20',
            'notes': 'Demonstration model - replace with real trained weights'
        }
    }

def main():
    print("🏗️  Creating real PyTorch OCR model...")
    
    # Create vocabulary
    vocab = create_vocabulary()
    vocab_size = vocab['vocab_size']
    print(f"📚 Created vocabulary with {vocab_size} characters")
    
    # Create model
    model = SimpleOCRModel(vocab_size=vocab_size)
    print(f"🤖 Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create model config
    config = create_model_config()
    config['vocab_size'] = vocab_size
    
    # Create temporary directory for model files
    temp_dir = Path('temp_real_model')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save model as TorchScript for better deployment
        model.eval()
        example_input = torch.randn(1, 3, 64, 256)  # Example input
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # Save files
        model_path = temp_dir / 'model.pth'
        torch.jit.save(traced_model, model_path)
        print(f"💾 Saved TorchScript model to {model_path}")
        
        # Save vocabulary
        vocab_path = temp_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"📖 Saved vocabulary to {vocab_path}")
        
        # Save config
        config_path = temp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"⚙️  Saved config to {config_path}")
        
        # Copy enhanced inference script
        import shutil
        inference_src = Path('scripts/inference_enhanced.py')
        inference_dst = temp_dir / 'inference.py'
        if inference_src.exists():
            shutil.copy2(inference_src, inference_dst)
            print(f"📋 Copied enhanced inference script to {inference_dst}")
        else:
            print("⚠️  Enhanced inference script not found - using basic one")
            # Create a basic inference script
            basic_inference = '''
# Basic inference script - replace with enhanced version
import torch
import json

def model_fn(model_dir):
    model_path = f"{model_dir}/model.pth"
    model = torch.jit.load(model_path, map_location='cpu')
    return model

def predict_fn(input_data, model):
    # Placeholder prediction
    return {"text": "Model loaded successfully", "confidence": 0.95}
'''
            with open(inference_dst, 'w') as f:
                f.write(basic_inference)
        
        # Create the model.tar.gz
        print("📦 Creating model.tar.gz...")
        with tarfile.open('model.tar.gz', 'w:gz') as tar:
            for file_path in temp_dir.glob('*'):
                tar.add(file_path, arcname=file_path.name)
        
        # Get file size
        tar_size = os.path.getsize('model.tar.gz')
        print(f"✅ Created model.tar.gz ({tar_size:,} bytes)")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print("\n🎉 Real PyTorch OCR model created successfully!")
        print("\n📋 What's included:")
        print("   ✅ TorchScript model (model.pth)")
        print("   ✅ Character vocabulary (vocab.json)")
        print("   ✅ Model configuration (config.json)")
        print("   ✅ Inference script (inference.py)")
        print("\n🔄 Next steps:")
        print("   1. This model has random weights - train it on real data")
        print("   2. The model will work with both Tesseract and PyTorch methods")
        print("   3. Deploy using: terraform apply")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        # Clean up on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

if __name__ == "__main__":
    main()
