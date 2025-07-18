#!/usr/bin/env python3
"""
Data preprocessing script for OCR pipeline
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--train-output-path', type=str, default='/opt/ml/processing/output/train')
    parser.add_argument('--validation-output-path', type=str, default='/opt/ml/processing/output/validation')
    parser.add_argument('--test-size', type=float, default=0.2)
    
    return parser.parse_args()

def main():
    """Main preprocessing function"""
    args = parse_args()
    
    logger.info("Starting data preprocessing...")
    
    # Create output directories
    os.makedirs(args.train_output_path, exist_ok=True)
    os.makedirs(args.validation_output_path, exist_ok=True)
    
    # For this example, we'll create dummy data
    # In a real scenario, you'd process actual data from the input path
    
    # Create dummy training data
    train_data = []
    val_data = []
    
    for i in range(80):  # 80 training samples
        train_data.append({
            'image_path': f'train_image_{i}.jpg',
            'text': f'Training sample text {i}'
        })
    
    for i in range(20):  # 20 validation samples
        val_data.append({
            'image_path': f'val_image_{i}.jpg',
            'text': f'Validation sample text {i}'
        })
    
    # Save as JSON files
    with open(os.path.join(args.train_output_path, 'data.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(args.validation_output_path, 'data.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    logger.info(f"Preprocessing completed. Created {len(train_data)} training and {len(val_data)} validation samples")

if __name__ == '__main__':
    main()
