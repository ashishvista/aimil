#!/usr/bin/env python3
"""
Simplified SageMaker inference script for OCR
This version handles missing dependencies gracefully
"""

import os
import json
import logging
import io
import base64
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Load the model. For simplicity, we'll just return a status dict.
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Check what's available
    status = {
        'model_dir': model_dir,
        'tesseract_available': False,
        'pil_available': False,
        'torch_available': False
    }
    
    # Check for PIL
    try:
        from PIL import Image
        status['pil_available'] = True
        logger.info("✅ PIL is available")
    except ImportError:
        logger.warning("❌ PIL not available")
    
    # Check for Tesseract binary
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, check=True)
        status['tesseract_available'] = True
        logger.info("✅ Tesseract is available")
        logger.info(f"Tesseract version: {result.stdout.split()[1] if result.stdout else 'unknown'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("❌ Tesseract binary not found")
    
    # Check for PyTorch
    try:
        import torch
        status['torch_available'] = True
        logger.info("✅ PyTorch is available")
    except ImportError:
        logger.warning("❌ PyTorch not available")
    
    return status

def input_fn(request_body, request_content_type):
    """
    Parse input data for inference.
    """
    logger.info(f"Content type: {request_content_type}")
    
    try:
        # Import PIL here to handle missing dependency
        from PIL import Image
        
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            
            if 'image' in input_data:
                # Base64 encoded image
                image_data = base64.b64decode(input_data['image'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                return {
                    'image': image, 
                    'method': input_data.get('method', 'tesseract'),
                    'image_size': len(image_data)
                }
            else:
                raise ValueError("No image data found in request")
        
        elif request_content_type in ['image/jpeg', 'image/png']:
            # Direct image upload
            image = Image.open(io.BytesIO(request_body)).convert('RGB')
            return {
                'image': image, 
                'method': 'tesseract',
                'image_size': len(request_body)
            }
        
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
            
    except ImportError:
        logger.error("PIL not available - cannot process images")
        raise ValueError("Image processing not available - PIL missing")

def predict_fn(input_data, model_dict):
    """
    Run inference on the input data.
    """
    logger.info(f"Starting inference with method: {input_data.get('method', 'tesseract')}")
    
    image = input_data['image']
    method = input_data.get('method', 'tesseract')
    
    results = {
        'processing_info': {
            'method_requested': method,
            'image_size': input_data.get('image_size', 0),
            'image_dimensions': f"{image.width}x{image.height}",
            'model_status': model_dict
        }
    }
    
    # Method 1: Tesseract OCR
    if method in ['tesseract', 'both']:
        if model_dict.get('tesseract_available', False):
            try:
                import pytesseract
                
                logger.info("Running Tesseract OCR...")
                
                # Basic OCR
                full_text = pytesseract.image_to_string(image, lang='eng')
                logger.info(f"Tesseract extracted text length: {len(full_text)}")
                
                # Get detailed word data
                word_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng')
                
                # Extract word boxes with confidence > 30
                word_boxes = []
                n_boxes = len(word_data['level'])
                for i in range(n_boxes):
                    confidence = int(word_data['conf'][i])
                    text = str(word_data['text'][i]).strip()
                    
                    if confidence > 30 and text:  # Only include confident, non-empty text
                        word_boxes.append({
                            'text': text,
                            'confidence': confidence,
                            'left': int(word_data['left'][i]),
                            'top': int(word_data['top'][i]),
                            'width': int(word_data['width'][i]),
                            'height': int(word_data['height'][i])
                        })
                
                results['tesseract'] = {
                    'success': True,
                    'full_text': full_text.strip(),
                    'word_boxes': word_boxes,
                    'total_words': len(word_boxes)
                }
                
                logger.info(f"Tesseract found {len(word_boxes)} confident words")
                
            except ImportError:
                logger.error("pytesseract module not available")
                results['tesseract'] = {
                    'success': False,
                    'error': 'pytesseract module not found',
                    'full_text': '',
                    'word_boxes': []
                }
            except Exception as e:
                logger.error(f"Tesseract OCR failed: {str(e)}")
                results['tesseract'] = {
                    'success': False,
                    'error': str(e),
                    'full_text': '',
                    'word_boxes': []
                }
        else:
            logger.warning("Tesseract binary not available")
            results['tesseract'] = {
                'success': False,
                'error': 'Tesseract binary not found on system',
                'full_text': 'Sample OCR output (Tesseract unavailable)',
                'word_boxes': [{
                    'text': 'Sample',
                    'confidence': 90,
                    'left': 10,
                    'top': 10,
                    'width': 100,
                    'height': 30
                }]
            }
    
    # Method 2: PyTorch model (placeholder)
    if method in ['pytorch', 'both']:
        if model_dict.get('torch_available', False):
            try:
                logger.info("PyTorch is available but model not implemented yet")
                results['pytorch'] = {
                    'success': False,
                    'predicted_text': 'PyTorch model not implemented',
                    'confidence': 0.0,
                    'error': 'PyTorch inference not yet implemented'
                }
            except Exception as e:
                logger.error(f"PyTorch inference failed: {str(e)}")
                results['pytorch'] = {
                    'success': False,
                    'error': str(e),
                    'predicted_text': '',
                    'confidence': 0.0
                }
        else:
            logger.warning("PyTorch not available")
            results['pytorch'] = {
                'success': False,
                'error': 'PyTorch not available',
                'predicted_text': '',
                'confidence': 0.0
            }
    
    # Add image metadata
    results['image_info'] = {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': getattr(image, 'format', 'Unknown')
    }
    
    logger.info("Inference completed successfully")
    return results

def output_fn(prediction, accept):
    """
    Format the prediction output.
    """
    logger.info(f"Formatting output for content type: {accept}")
    
    # Handle various accept types gracefully
    if accept in ['application/json', '*/*', None]:
        return json.dumps(prediction, indent=2), 'application/json'
    elif accept == 'text/plain':
        # Return just the extracted text for text/plain requests
        text_content = ""
        if 'tesseract' in prediction and prediction['tesseract'].get('success'):
            text_content = prediction['tesseract']['full_text']
        elif 'pytorch' in prediction and prediction['pytorch'].get('success'):
            text_content = prediction['pytorch']['predicted_text']
        else:
            text_content = "No text could be extracted"
        return text_content, 'text/plain'
    else:
        # Default to JSON for any unrecognized accept type
        logger.warning(f"Unrecognized accept type '{accept}', defaulting to application/json")
        return json.dumps(prediction, indent=2), 'application/json'
