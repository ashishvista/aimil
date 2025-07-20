#!/usr/bin/env python3
"""
Test script to verify the permissions-fixed SageMaker endpoint
"""

import json
import base64
import boto3
from botocore.exceptions import ClientError
import time

def test_endpoint():
    """Test the OCR SageMaker endpoint"""
    
    # Configuration
    endpoint_name = "ocr-pipeline-endpoint-sv1mmdug"
    region = "ap-south-1"
    profile = "test-prod"
    
    # Initialize SageMaker runtime client
    session = boto3.Session(profile_name=profile)
    sagemaker_runtime = session.client('sagemaker-runtime', region_name=region)
    
    # Sample text image (base64 encoded simple text)
    # This is a simple white image with black text "HELLO WORLD"
    sample_image_b64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCABQAeADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/KKKKAP/2Q=="
    
    # Prepare the payload
    payload = {
        "image": sample_image_b64,
        "content_type": "image/jpeg"
    }
    
    try:
        print(f"🔍 Testing endpoint: {endpoint_name}")
        print(f"📍 Region: {region}")
        
        # Call the endpoint
        print("📤 Sending request to endpoint...")
        start_time = time.time()
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        end_time = time.time()
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        print(f"✅ Request successful!")
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📄 Response:")
        print(json.dumps(result, indent=2))
        
        # Check if OCR text was extracted
        if 'extracted_text' in result:
            if result['extracted_text'].strip():
                print(f"🎉 OCR extraction successful!")
                print(f"📝 Extracted text: '{result['extracted_text'].strip()}'")
            else:
                print("⚠️  OCR extraction returned empty text")
        else:
            print("⚠️  No 'extracted_text' in response")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"❌ AWS Error: {error_code}")
        print(f"💬 Message: {error_message}")
        
        if error_code == 'ModelError':
            print("🔧 This suggests an issue with the model container")
        elif error_code == 'ValidationException':
            print("🔧 This suggests an issue with the request format")
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def check_endpoint_status():
    """Check the current status of the endpoint"""
    
    endpoint_name = "ocr-pipeline-endpoint-sv1mmdug"
    region = "ap-south-1"
    profile = "test-prod"
    
    session = boto3.Session(profile_name=profile)
    sagemaker = session.client('sagemaker', region_name=region)
    
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        
        status = response['EndpointStatus']
        print(f"📊 Endpoint Status: {status}")
        
        if status == 'InService':
            print("✅ Endpoint is ready for testing!")
            return True
        elif status == 'Creating':
            print("⏳ Endpoint is still being created...")
            return False
        elif status == 'Failed':
            print("❌ Endpoint creation failed!")
            if 'FailureReason' in response:
                print(f"💬 Failure reason: {response['FailureReason']}")
            return False
        else:
            print(f"⚠️  Endpoint status: {status}")
            return False
            
    except ClientError as e:
        print(f"❌ Error checking endpoint status: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing OCR SageMaker Endpoint with Permissions Fix")
    print("=" * 60)
    
    # First check if endpoint is ready
    if check_endpoint_status():
        print("\n" + "=" * 60)
        test_endpoint()
    else:
        print("\n⏳ Please wait for the endpoint to be ready and run this test again.")
        print("💡 You can check the status using: aws sagemaker describe-endpoint --endpoint-name ocr-pipeline-endpoint-sv1mmdug --profile test-prod --region ap-south-1")
