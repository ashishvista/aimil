#!/usr/bin/env python3
"""
Test script for SageMaker OCR container functionality
"""
import os
import sys

def test_imports():
    """Test all required package imports"""
    print("🔍 Testing OCR component imports...")
    
    try:
        import pytesseract
        print(f"✅ PyTesseract version: {pytesseract.__version__}")
    except ImportError as e:
        print(f"❌ PyTesseract import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ PIL version: {Image.__version__}")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    return True

def test_tesseract_binary():
    """Test Tesseract binary accessibility"""
    print("\n🔧 Testing Tesseract binary...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract binary version: {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract binary error: {e}")
        return False

def test_sagemaker_directories():
    """Test SageMaker directory structure and permissions"""
    print("\n📁 Testing SageMaker directories...")
    
    required_dirs = [
        '/opt/ml/code',
        '/opt/ml/model', 
        '/opt/ml/input',
        '/opt/ml/output',
        '/home/model-server/.sagemaker'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            stat_info = os.stat(dir_path)
            permissions = oct(stat_info.st_mode)[-3:]
            print(f"✅ {dir_path} (permissions: {permissions})")
        else:
            print(f"❌ Missing: {dir_path}")
            all_good = False
    
    return all_good

def test_inference_script():
    """Test if inference script exists"""
    print("\n📄 Testing inference script...")
    
    script_path = '/opt/ml/code/inference.py'
    if os.path.exists(script_path):
        print(f"✅ Inference script found: {script_path}")
        return True
    else:
        print(f"❌ Missing inference script: {script_path}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting SageMaker OCR Container Tests\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Tesseract Binary", test_tesseract_binary), 
        ("SageMaker Directories", test_sagemaker_directories),
        ("Inference Script", test_inference_script)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Container is ready for SageMaker!")
    else:
        print("⚠️  Some tests failed - Review issues above")
    print("="*50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
