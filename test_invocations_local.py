#!/usr/bin/env python3
"""
Test script for the /invocations endpoint of our local Docker container
"""
import requests
import json
import base64
import time
from PIL import Image
import io

def test_invocations_endpoint():
    """Test the /invocations endpoint with sample data"""
    
    # Create a simple test image (white background with black text)
    img = Image.new('RGB', (400, 100), color='white')
    
    # Convert to base64
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Test data - same format as SageMaker would send
    test_data = {
        "instances": [
            {
                "image": img_base64,
                "format": "png"
            }
        ]
    }
    
    url = "http://localhost:8080/invocations"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    print("🔍 Testing /invocations endpoint...")
    print(f"URL: {url}")
    print(f"Payload size: {len(json.dumps(test_data))} bytes")
    
    try:
        # Make the request
        response = requests.post(url, 
                               data=json.dumps(test_data),
                               headers=headers,
                               timeout=30)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS! Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ ERROR Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")

def test_with_actual_image():
    """Test with the actual test image we have"""
    try:
        with open('/Users/ashish_kumar/aiml/test_image.png', 'rb') as f:
            img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        test_data = {
            "instances": [
                {
                    "image": img_base64,
                    "format": "png"
                }
            ]
        }
        
        url = "http://localhost:8080/invocations"
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        print("\n🖼️ Testing with actual test image...")
        
        response = requests.post(url, 
                               data=json.dumps(test_data),
                               headers=headers,
                               timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS! OCR Result: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ ERROR Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing with actual image: {e}")

def test_health_check():
    """Test the health check endpoint"""
    try:
        print("\n❤️ Testing health check...")
        response = requests.get("http://localhost:8080/ping", timeout=10)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health check passed!")
        else:
            print(f"❌ Health check failed: {response.text}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🚀 Starting local Docker container tests...\n")
    
    # Wait a moment for container to be fully ready
    print("⏳ Waiting 2 seconds for container to be ready...")
    time.sleep(2)
    
    # Test health check first
    test_health_check()
    
    # Test with simple image
    test_invocations_endpoint()
    
    # Test with actual image
    test_with_actual_image()
    
    print("\n🏁 Test completed!")
