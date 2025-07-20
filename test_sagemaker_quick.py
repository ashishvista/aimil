#!/usr/bin/env python3
"""
Quick SageMaker endpoint test script
Simple test to verify if the endpoint is working
"""

import boto3
import json
import base64
import sys
from datetime import datetime

def quick_test():
    """Quick test of the SageMaker endpoint"""
    print("🚀 Quick SageMaker Endpoint Test")
    print("=" * 40)
    
    # Configuration
    profile = 'test-prod'
    region = 'ap-south-1'
    
    try:
        # Initialize AWS clients
        session = boto3.Session(profile_name=profile)
        sagemaker_runtime = session.client('sagemaker-runtime', region_name=region)
        sagemaker_client = session.client('sagemaker', region_name=region)
        print(f"✅ AWS clients initialized")
        
        # Get endpoint name from terraform
        import subprocess
        result = subprocess.run(['terraform', 'output', '-raw', 'sagemaker_endpoint_name'], 
                              capture_output=True, text=True, cwd='/Users/ashish_kumar/aiml')
        endpoint_name = result.stdout.strip()
        print(f"✅ Endpoint name: {endpoint_name}")
        
        # Check endpoint status
        endpoint_info = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = endpoint_info['EndpointStatus']
        print(f"✅ Endpoint status: {status}")
        
        if status != 'InService':
            print(f"❌ Endpoint is not InService. Current status: {status}")
            return False
        
        # Load test image base64
        try:
            with open('/Users/ashish_kumar/aiml/test_image_b64.txt', 'r') as f:
                image_base64 = f.read().strip()
            print(f"✅ Test image loaded: {len(image_base64)} characters")
        except Exception as e:
            print(f"❌ Failed to load test image: {e}")
            return False
        
        # Prepare payload
        payload = {
            'image': image_base64,
            'method': 'tesseract'
        }
        
        # Test endpoint
        print("🔄 Testing endpoint...")
        start_time = datetime.now()
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Parse response
        response_body = response['Body'].read().decode()
        result = json.loads(response_body)
        
        print(f"✅ Test completed in {duration:.2f} seconds")
        
        # Extract text result
        if 'extracted_text' in result:
            extracted_text = result['extracted_text']
        elif 'text' in result:
            extracted_text = result['text']
        else:
            extracted_text = str(result)
        
        print(f"📝 OCR Result: '{extracted_text.strip()}'")
        print("🎉 Quick test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
