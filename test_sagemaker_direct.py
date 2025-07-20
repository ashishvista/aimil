#!/usr/bin/env python3
"""
Direct SageMaker endpoint test script
Tests the OCR endpoint directly without Lambda wrapper
"""

import boto3
import json
import base64
import sys
import os
from datetime import datetime

# Initialize SageMaker runtime client
session = boto3.Session(profile_name='test-prod')
sagemaker_runtime = session.client('sagemaker-runtime', region_name='ap-south-1')

def test_sagemaker_endpoint():
    print("🚀 Starting direct SageMaker endpoint test")
    print(f"⏰ Test started at: {datetime.now()}")
    
    # Get endpoint name from terraform output
    print("🔄 Getting endpoint name from terraform...")
    try:
        import subprocess
        result = subprocess.run(['terraform', 'output', '-raw', 'sagemaker_endpoint_name'], 
                              capture_output=True, text=True, cwd='/Users/ashish_kumar/aiml')
        endpoint_name = result.stdout.strip()
        print(f"✅ Endpoint name: {endpoint_name}")
    except Exception as e:
        print(f"❌ Failed to get endpoint name: {e}")
        return False
    
    # Load test image
    image_path = '/Users/ashish_kumar/aiml/test_image.png'
    print(f"🔄 Loading test image: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        print(f"✅ Image loaded successfully, size: {len(image_data)} bytes")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return False
    
    # Prepare payload
    print("🔄 Preparing payload...")
    try:
        payload = {
            'image': base64.b64encode(image_data).decode('utf-8'),
            'method': 'tesseract'
        }
        payload_json = json.dumps(payload)
        print(f"✅ Payload prepared, size: {len(payload_json)} bytes")
        print(f"📊 Base64 image size: {len(payload['image'])} characters")
    except Exception as e:
        print(f"❌ Failed to prepare payload: {e}")
        return False
    
    # Test 1: Check endpoint status first
    print("\n📋 Test 1: Checking endpoint status...")
    try:
        sm_client = session.client('sagemaker', region_name='ap-south-1')
        endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = endpoint_info['EndpointStatus']
        print(f"✅ Endpoint status: {status}")
        
        if status != 'InService':
            print(f"❌ Endpoint is not InService, current status: {status}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check endpoint status: {e}")
        return False
    
    # Test 2: Invoke endpoint with short timeout first
    print("\n📋 Test 2: Testing with short timeout (30s)...")
    try:
        print("🔄 Invoking SageMaker endpoint...")
        start_time = datetime.now()
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload_json,
            Accept='application/json'
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"✅ Endpoint invoked successfully in {duration:.2f} seconds")
        
        # Parse response
        response_body = response['Body'].read().decode()
        print(f"📤 Raw response length: {len(response_body)} characters")
        
        try:
            result = json.loads(response_body)
            print("✅ Response parsed as JSON")
            print(f"📋 Response keys: {list(result.keys())}")
            
            # Print the OCR result
            if 'extracted_text' in result:
                print(f"📝 Extracted text: '{result['extracted_text']}'")
            elif 'text' in result:
                print(f"📝 Extracted text: '{result['text']}'")
            elif 'result' in result:
                print(f"📝 Result: {result['result']}")
            else:
                print(f"📝 Full response: {json.dumps(result, indent=2)}")
                
        except json.JSONDecodeError:
            print(f"⚠️  Response is not JSON: {response_body[:500]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Endpoint invocation failed: {e}")
        print(f"💥 Error type: {type(e).__name__}")
        
        # Check if it's a timeout
        if 'timeout' in str(e).lower() or 'time' in str(e).lower():
            print("⏰ This appears to be a timeout error")
        
        return False

if __name__ == "__main__":
    print("🧪 SageMaker Direct Endpoint Test")
    print("=" * 50)
    
    success = test_sagemaker_endpoint()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
