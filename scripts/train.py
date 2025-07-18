#!/usr/bin/env python3
"""
SageMaker training script for OCR model
This script handles the training of an OCR model using PyTorch and Tesseract
"""

import os
import sys
import json
import argparse
import logging
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytesseract
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRDataset(Dataset):
    """Custom dataset for OCR training"""
    
    def __init__(self, image_paths, texts, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, text

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--vocab-size', type=int, default=1000)
    
    return parser.parse_args()

def create_vocab_from_texts(texts, vocab_size=1000):
    """Create vocabulary from text data"""
    char_counts = {}
    for text in texts:
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
    
    # Sort by frequency and take top vocab_size
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {char: idx for idx, (char, _) in enumerate(sorted_chars[:vocab_size])}
    vocab['<UNK>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    
    return vocab

def text_to_sequence(text, vocab, max_length=100):
    """Convert text to sequence of token IDs"""
    sequence = [vocab.get(char, vocab['<UNK>']) for char in text]
    
    # Pad or truncate to max_length
    if len(sequence) < max_length:
        sequence.extend([vocab['<PAD>']] * (max_length - len(sequence)))
    else:
        sequence = sequence[:max_length]
    
    return sequence

def train_model(model, train_loader, val_loader, device, args):
    """Train the OCR model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, texts) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # For this simplified example, we'll use a dummy loss
            # In a real OCR model, you'd need proper sequence-to-sequence training
            dummy_targets = torch.randint(0, args.vocab_size, (images.size(0),)).to(device)
            loss = criterion(outputs, dummy_targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device)
                outputs = model(images)
                dummy_targets = torch.randint(0, args.vocab_size, (images.size(0),)).to(device)
                loss = criterion(outputs, dummy_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f'Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
        
        model.train()

def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # For this example, we'll create dummy data
    # In a real scenario, you'd load actual image and text data
    logger.info("Creating dummy training data...")
    
    # Create dummy image paths and texts
    dummy_images = []
    dummy_texts = []
    
    for i in range(100):  # Create 100 dummy samples
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        image_path = os.path.join('/tmp', f'dummy_image_{i}.jpg')
        dummy_image.save(image_path)
        dummy_images.append(image_path)
        dummy_texts.append(f"Sample text {i}")
    
    # Split into train and validation
    train_images, val_images, train_texts, val_texts = train_test_split(
        dummy_images, dummy_texts, test_size=0.2, random_state=42
    )
    
    # Create vocabulary
    vocab = create_vocab_from_texts(train_texts + val_texts, args.vocab_size)
    
    # Save vocabulary
    with open(os.path.join(args.model_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = OCRDataset(train_images, train_texts, transform=transform)
    val_dataset = OCRDataset(val_images, val_texts, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = SimpleOCRModel(vocab_size=args.vocab_size, hidden_dim=args.hidden_dim)
    model.to(device)
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_model(model, train_loader, val_loader, device, args)
    
    # Save model configuration
    model_config = {
        'vocab_size': args.vocab_size,
        'hidden_dim': args.hidden_dim,
        'model_class': 'SimpleOCRModel'
    }
    
    with open(os.path.join(args.model_dir, 'config.json'), 'w') as f:
        json.dump(model_config, f)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()
