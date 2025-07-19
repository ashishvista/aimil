#!/usr/bin/env python3
"""
Test script to verify Lambda layer dependencies work correctly
"""

import sys
import os

# Add the layer path to the Python path (simulating Lambda environment)
layer_path = os.path.join(os.path.dirname(__file__), 'lambda_layer_temp', 'python')
if os.path.exists(layer_path):
    sys.path.insert(0, layer_path)

def test_imports():
    """Test importing the required packages"""
    print("🧪 Testing Lambda layer imports...")
    
    try:
        import PIL
        print("✅ PIL import successful")
        print(f"   PIL version: {PIL.__version__}")
        print(f"   PIL path: {PIL.__file__}")
        
        from PIL import Image
        print("✅ PIL.Image import successful")
        
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests import successful")
        print(f"   requests version: {requests.__version__}")
        print(f"   requests path: {requests.__file__}")
        
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    print("🎉 All imports successful!")
    return True

if __name__ == "__main__":
    # Extract the layer to test locally
    if not os.path.exists('lambda_layer_temp'):
        print("📦 Extracting lambda layer for testing...")
        os.system('unzip -q lambda_layer.zip -d lambda_layer_temp')
    
    success = test_imports()
    
    # Clean up
    if os.path.exists('lambda_layer_temp'):
        os.system('rm -rf lambda_layer_temp')
    
    sys.exit(0 if success else 1)
