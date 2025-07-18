#!/usr/bin/env python3
"""
SageMaker inference script for OCR model
This script handles model loading and inference for the deployed OCR model
"""

import os
import json
import logging
import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pytesseract
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Load model configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load vocabulary
    vocab_path = os.path.join(model_dir, 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Create model
    model = SimpleOCRModel(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim']
    )
    
    # Load model weights
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Return model and vocab for use in prediction
    return {'model': model, 'vocab': vocab, 'reverse_vocab': {v: k for k, v in vocab.items()}}

def input_fn(request_body, request_content_type):
    """
    Parse input data for inference.
    """
    logger.info(f"Content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        if 'image' in input_data:
            # Base64 encoded image
            image_data = base64.b64decode(input_data['image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return {'image': image, 'method': input_data.get('method', 'tesseract')}
        else:
            raise ValueError("No image data found in request")
    
    elif request_content_type == 'image/jpeg' or request_content_type == 'image/png':
        # Direct image upload
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        return {'image': image, 'method': 'tesseract'}
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """
    Run inference on the input data.
    """
    image = input_data['image']
    method = input_data.get('method', 'tesseract')
    
    results = {}
    
    # Method 1: Tesseract OCR (traditional)
    if method in ['tesseract', 'both']:
        try:
            # Use Tesseract for OCR
            tesseract_text = pytesseract.image_to_string(image)
            tesseract_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract bounding boxes and confidence scores
            boxes = []
            for i in range(len(tesseract_data['text'])):
                if int(tesseract_data['conf'][i]) > 0:  # Only include confident detections
                    box = {
                        'text': tesseract_data['text'][i],
                        'confidence': float(tesseract_data['conf'][i]) / 100.0,
                        'bbox': {
                            'left': int(tesseract_data['left'][i]),
                            'top': int(tesseract_data['top'][i]),
                            'width': int(tesseract_data['width'][i]),
                            'height': int(tesseract_data['height'][i])
                        }
                    }
                    boxes.append(box)
            
            results['tesseract'] = {
                'full_text': tesseract_text.strip(),
                'word_boxes': boxes
            }
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            results['tesseract'] = {
                'error': str(e),
                'full_text': '',
                'word_boxes': []
            }
    
    # Method 2: Custom PyTorch model (for demonstration)
    if method in ['pytorch', 'both']:
        try:
            model = model_dict['model']
            vocab = model_dict['vocab']
            reverse_vocab = model_dict['reverse_vocab']
            
            # Preprocess image for PyTorch model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor_image = transform(image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = model(tensor_image)
                predictions = torch.softmax(outputs, dim=1)
                top_predictions = torch.topk(predictions, 5, dim=1)
            
            # Convert predictions to text (simplified)
            predicted_chars = []
            for idx in top_predictions.indices[0]:
                if idx.item() in reverse_vocab:
                    predicted_chars.append(reverse_vocab[idx.item()])
            
            results['pytorch'] = {
                'predicted_text': ''.join(predicted_chars[:10]),  # Take first 10 characters
                'confidence': float(top_predictions.values[0][0])
            }
        except Exception as e:
            logger.error(f"PyTorch model inference failed: {str(e)}")
            results['pytorch'] = {
                'error': str(e),
                'predicted_text': '',
                'confidence': 0.0
            }
    
    # Add image metadata
    results['image_info'] = {
        'width': image.width,
        'height': image.height,
        'mode': image.mode
    }
    
    return results

def output_fn(prediction, accept):
    """
    Format the prediction output.
    """
    if accept == 'application/json':
        return json.dumps(prediction, indent=2), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

def lambda_handler(event, context):
    """
    Lambda handler function for direct Lambda invocation (not SageMaker)
    """
    try:
        # This is a fallback handler for testing
        body = event.get('body', '')
        content_type = event.get('headers', {}).get('content-type', 'application/json')
        
        # Load model (in production, this would be cached)
        model_dict = model_fn('/opt/ml/model')
        
        # Parse input
        input_data = input_fn(body, content_type)
        
        # Run prediction
        result = predict_fn(input_data, model_dict)
        
        # Format output
        response_body, response_content_type = output_fn(result, 'application/json')
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': response_content_type,
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': response_body
        }
    
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }
