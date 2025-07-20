#!/usr/bin/env python3
"""
Enhanced SageMaker inference script for OCR with Tesseract support
This version works with custom Docker containers that have Tesseract installed
"""

import os
import json
import logging
import io
import base64
import subprocess
import sys
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model and check available dependencies.
    This function is called once when the container starts.
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Initialize status dictionary
    status = {
        'model_dir': model_dir,
        'tesseract_available': False,
        'pil_available': False,
        'torch_available': False,
        'opencv_available': False,
        'pytesseract_available': False
    }
    
    # Check for PIL/Pillow
    try:
        from PIL import Image, ImageEnhance
        status['pil_available'] = True
        logger.info("✅ PIL/Pillow is available")
    except ImportError as e:
        logger.warning(f"❌ PIL/Pillow not available: {e}")
    
    # Check for Tesseract binary
    try:
        result = subprocess.run(
            ['tesseract', '--version'], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=10
        )
        status['tesseract_available'] = True
        version_line = result.stdout.split('\n')[0] if result.stdout else 'unknown'
        logger.info(f"✅ Tesseract binary available: {version_line}")
        
        # Also check installed languages
        lang_result = subprocess.run(
            ['tesseract', '--list-langs'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if lang_result.returncode == 0:
            languages = lang_result.stdout.strip().split('\n')[1:]  # Skip header
            logger.info(f"Available Tesseract languages: {', '.join(languages[:5])}...")
            status['tesseract_languages'] = languages
        
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"❌ Tesseract binary not found: {e}")
        status['tesseract_languages'] = []
    
    # Check for pytesseract wrapper
    try:
        import pytesseract
        status['pytesseract_available'] = True
        logger.info("✅ pytesseract wrapper is available")
        
        # Set Tesseract path if available
        if status['tesseract_available']:
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
            
    except ImportError as e:
        logger.warning(f"❌ pytesseract wrapper not available: {e}")
    
    # Check for OpenCV
    try:
        import cv2
        status['opencv_available'] = True
        logger.info(f"✅ OpenCV is available (version: {cv2.__version__})")
    except ImportError as e:
        logger.warning(f"❌ OpenCV not available: {e}")
    
    # Check for PyTorch
    try:
        import torch
        status['torch_available'] = True
        logger.info(f"✅ PyTorch is available (version: {torch.__version__})")
        
        # Load the actual PyTorch model if available
        model_path = os.path.join(model_dir, 'model.pth')
        if os.path.exists(model_path):
            try:
                torch_model = torch.jit.load(model_path, map_location='cpu')
                torch_model.eval()
                status['torch_model'] = torch_model
                logger.info(f"✅ PyTorch model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load PyTorch model: {e}")
                status['torch_model'] = None
        else:
            status['torch_model'] = None
            logger.info("No PyTorch model file found - OCR will use other methods")
            
    except ImportError as e:
        logger.warning(f"❌ PyTorch not available: {e}")
    
    logger.info(f"Model loading complete. Available methods: "
                f"Tesseract={status['tesseract_available']}, "
                f"PyTorch={status['torch_available']}, "
                f"OpenCV={status['opencv_available']}")
    
    return status

def input_fn(request_body: Union[str, bytes], request_content_type: str) -> Dict[str, Any]:
    """
    Parse input data for inference.
    """
    logger.info(f"Processing request with content type: {request_content_type}")
    
    try:
        from PIL import Image
        
        if request_content_type == 'application/json':
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
                
            input_data = json.loads(request_body)
            logger.info(f"JSON request keys: {list(input_data.keys())}")
            
            if 'image' in input_data:
                # Base64 encoded image in 'image' field
                image_data = base64.b64decode(input_data['image'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                return {
                    'image': image,
                    'method': input_data.get('method', 'tesseract'),
                    'image_size': len(image_data),
                    'options': input_data.get('options', {})
                }
            else:
                raise ValueError("No image data found in request. Expected 'image' field with base64 data.")
        
        elif request_content_type in ['image/jpeg', 'image/png', 'image/jpg']:
            # Direct image upload
            if isinstance(request_body, str):
                # If it's a string, assume it's base64
                image_data = base64.b64decode(request_body)
            else:
                image_data = request_body
                
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return {
                'image': image,
                'method': 'tesseract',
                'image_size': len(image_data),
                'options': {}
            }
        
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
            
    except ImportError:
        logger.error("PIL not available - cannot process images")
        raise ValueError("Image processing not available - PIL missing")
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise ValueError(f"Error parsing input: {e}")

def predict_fn(input_data: Dict[str, Any], model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on the input data using available OCR methods.
    """
    image = input_data['image']
    method = input_data.get('method', 'tesseract')
    options = input_data.get('options', {})
    
    logger.info(f"Running inference with method: {method} on image size: {image.width}x{image.height}")
    
    results = {
        'processing_info': {
            'method_requested': method,
            'image_size': input_data.get('image_size', 0),
            'image_dimensions': f"{image.width}x{image.height}",
            'model_status': {k: v for k, v in model_dict.items() if k != 'torch_model'}
        }
    }
    
    # Method 1: Tesseract OCR
    if method in ['tesseract', 'both', 'auto']:
        if model_dict.get('tesseract_available', False) and model_dict.get('pytesseract_available', False):
            try:
                import pytesseract
                
                logger.info("Running Tesseract OCR...")
                
                # OCR configuration
                lang = options.get('language', 'eng')
                config = options.get('config', '--oem 3 --psm 6')
                
                # Basic OCR
                full_text = pytesseract.image_to_string(image, lang=lang, config=config)
                logger.info(f"Tesseract extracted {len(full_text)} characters")
                
                # Get detailed word data
                try:
                    word_data = pytesseract.image_to_data(
                        image, 
                        output_type=pytesseract.Output.DICT, 
                        lang=lang,
                        config=config
                    )
                    
                    # Extract word boxes with confidence threshold
                    confidence_threshold = options.get('confidence_threshold', 30)
                    word_boxes = []
                    
                    n_boxes = len(word_data['level'])
                    for i in range(n_boxes):
                        confidence = int(word_data['conf'][i])
                        text = str(word_data['text'][i]).strip()
                        
                        if confidence > confidence_threshold and text:
                            word_boxes.append({
                                'text': text,
                                'confidence': confidence,
                                'bbox': {
                                    'left': int(word_data['left'][i]),
                                    'top': int(word_data['top'][i]),
                                    'width': int(word_data['width'][i]),
                                    'height': int(word_data['height'][i])
                                }
                            })
                    
                    results['tesseract'] = {
                        'success': True,
                        'full_text': full_text.strip(),
                        'word_boxes': word_boxes,
                        'total_words': len(word_boxes),
                        'language': lang,
                        'config': config
                    }
                    
                    logger.info(f"Tesseract found {len(word_boxes)} confident words")
                    
                except Exception as detail_error:
                    logger.warning(f"Could not extract detailed word data: {detail_error}")
                    results['tesseract'] = {
                        'success': True,
                        'full_text': full_text.strip(),
                        'word_boxes': [],
                        'total_words': 0,
                        'language': lang,
                        'warning': str(detail_error)
                    }
                
            except ImportError:
                logger.error("pytesseract module not available despite being detected")
                results['tesseract'] = {
                    'success': False,
                    'error': 'pytesseract module import failed',
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
            logger.warning("Tesseract not fully available")
            results['tesseract'] = {
                'success': False,
                'error': 'Tesseract binary or pytesseract wrapper not available',
                'full_text': 'Tesseract OCR not available in this container',
                'word_boxes': [],
                'available_methods': list(model_dict.keys())
            }
    
    # Method 2: PyTorch model
    if method in ['pytorch', 'torch', 'both', 'auto']:
        if model_dict.get('torch_available', False):
            torch_model = model_dict.get('torch_model')
            if torch_model is not None:
                try:
                    logger.info("Running PyTorch OCR inference...")
                    
                    # This is a placeholder - implement actual PyTorch OCR logic here
                    # For now, just return a success status
                    results['pytorch'] = {
                        'success': True,
                        'predicted_text': 'PyTorch OCR inference completed (placeholder)',
                        'confidence': 0.85,
                        'note': 'PyTorch model inference is not yet fully implemented'
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
                results['pytorch'] = {
                    'success': False,
                    'error': 'PyTorch model not loaded',
                    'predicted_text': '',
                    'confidence': 0.0
                }
        else:
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
        'format': getattr(image, 'format', None)
    }
    
    # Add processing summary
    successful_methods = [method for method in ['tesseract', 'pytorch'] 
                         if method in results and results[method].get('success', False)]
    
    results['summary'] = {
        'successful_methods': successful_methods,
        'primary_method': successful_methods[0] if successful_methods else None,
        'processing_time_info': 'Processing completed successfully'
    }
    
    logger.info(f"Inference completed. Successful methods: {successful_methods}")
    return results

def output_fn(prediction: Dict[str, Any], accept: str) -> tuple:
    """
    Format the prediction output based on the requested content type.
    """
    logger.info(f"Formatting output for content type: {accept}")
    
    # Handle various accept types gracefully
    if accept in ['application/json', '*/*', None]:
        return json.dumps(prediction, indent=2, default=str), 'application/json'
    elif accept == 'text/plain':
        # Return just the extracted text
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
        return json.dumps(prediction, indent=2, default=str), 'application/json'

if __name__ == "__main__":
    """
    Standalone HTTP server for testing and direct deployment
    """
    import argparse
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse
    
    parser = argparse.ArgumentParser(description='OCR Inference Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    # Load the model once at startup
    model_dir = os.environ.get('SAGEMAKER_MODEL_DIR', '/opt/ml/model')
    model = model_fn(model_dir)
    
    class OCRHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/ping':
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'OK')
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_POST(self):
            if self.path == '/invocations':
                try:
                    # Read the request body
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    
                    # Get content type from headers
                    content_type = self.headers.get('Content-Type', 'application/json')
                    
                    # Parse input using the SageMaker input_fn
                    parsed_input = input_fn(post_data, content_type)
                    
                    # Process the request using predict_fn
                    prediction = predict_fn(parsed_input, model)
                    
                    # Format the response
                    accept = self.headers.get('Accept', 'application/json')
                    response_data, content_type = output_fn(prediction, accept)
                    
                    # Send the response
                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.end_headers()
                    
                    if isinstance(response_data, str):
                        self.wfile.write(response_data.encode('utf-8'))
                    else:
                        self.wfile.write(response_data)
                        
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = json.dumps({'error': str(e)})
                    self.wfile.write(error_response.encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
    
    # Start the server
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, OCRHandler)
    
    logger.info(f"🚀 Starting OCR inference server on {args.host}:{args.port}")
    logger.info("Server is ready to handle requests")
    logger.info("Health check endpoint: /ping")
    logger.info("Inference endpoint: /invocations")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopping...")
        httpd.server_close()
