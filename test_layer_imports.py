#!/usr/bin/env python3
"""Test script to verify Lambda layer imports work correctly."""

import sys
import os

# Simulate Lambda layer structure
sys.path.insert(0, 'lambda_layer_temp/python' if os.path.exists('lambda_layer_temp/python') else '.')

def test_imports():
    """Test importing all required packages."""
    try:
        print("🧪 Testing package imports...")
        
        # Test PIL import
        print("📦 Testing PIL/Pillow...")
        from PIL import Image
        print("✅ PIL.Image imported successfully")
        
        # Test creating a simple image
        img = Image.new('RGB', (100, 100), color='red')
        print("✅ PIL image creation successful")
        
        # Test requests
        print("📦 Testing requests...")
        import requests
        print("✅ requests imported successfully")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
