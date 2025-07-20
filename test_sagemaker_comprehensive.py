#!/usr/bin/env python3
"""
Comprehensive SageMaker endpoint test script
Tests the OCR endpoint with multiple scenarios and methods
"""

import boto3
import json
import base64
import sys
import os
from datetime import datetime
import time
from botocore.exceptions import ClientError, NoCredentialsError

class SageMakerEndpointTester:
    def __init__(self, profile='test-prod', region='ap-south-1'):
        """Initialize the tester with AWS profile and region"""
        self.profile = profile
        self.region = region
        self.endpoint_name = None
        
        try:
            # Initialize AWS session and clients
            self.session = boto3.Session(profile_name=profile)
            self.sagemaker_runtime = self.session.client('sagemaker-runtime', region_name=region)
            self.sagemaker_client = self.session.client('sagemaker', region_name=region)
            print(f"✅ AWS session initialized with profile: {profile}, region: {region}")
        except NoCredentialsError:
            print(f"❌ No credentials found for profile: {profile}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to initialize AWS session: {e}")
            sys.exit(1)
    
    def get_endpoint_name(self):
        """Get endpoint name from terraform output or manual input"""
        print("🔄 Getting SageMaker endpoint name...")
        
        # Method 1: Try to get from terraform output
        try:
            import subprocess
            result = subprocess.run(['terraform', 'output', '-raw', 'sagemaker_endpoint_name'], 
                                  capture_output=True, text=True, cwd='/Users/ashish_kumar/aiml')
            if result.returncode == 0 and result.stdout.strip():
                self.endpoint_name = result.stdout.strip()
                print(f"✅ Got endpoint name from terraform: {self.endpoint_name}")
                return True
        except Exception as e:
            print(f"⚠️ Could not get endpoint name from terraform: {e}")
        
        # Method 2: List available endpoints
        try:
            endpoints = self.sagemaker_client.list_endpoints()
            ocr_endpoints = [ep for ep in endpoints['Endpoints'] if 'ocr' in ep['EndpointName'].lower()]
            
            if ocr_endpoints:
                self.endpoint_name = ocr_endpoints[0]['EndpointName']
                print(f"✅ Found OCR endpoint: {self.endpoint_name}")
                return True
            else:
                print("❌ No OCR endpoints found")
                return False
                
        except Exception as e:
            print(f"❌ Failed to list endpoints: {e}")
            return False
    
    def check_endpoint_status(self):
        """Check if endpoint is ready for inference"""
        print(f"\n📋 Checking endpoint status: {self.endpoint_name}")
        
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
            status = response['EndpointStatus']
            
            print(f"🔍 Endpoint Status: {status}")
            print(f"📅 Created: {response['CreationTime']}")
            print(f"🔄 Last Modified: {response['LastModifiedTime']}")
            
            if 'FailureReason' in response:
                print(f"❌ Failure Reason: {response['FailureReason']}")
            
            return status == 'InService'
            
        except ClientError as e:
            print(f"❌ Failed to describe endpoint: {e}")
            return False
    
    def load_test_data(self):
        """Load test image data in multiple formats"""
        test_data = {}
        
        # Method 1: Load from base64 file
        base64_file = '/Users/ashish_kumar/aiml/test_image_b64.txt'
        if os.path.exists(base64_file):
            try:
                with open(base64_file, 'r') as f:
                    test_data['from_base64_file'] = f.read().strip()
                print(f"✅ Loaded base64 from file: {len(test_data['from_base64_file'])} characters")
            except Exception as e:
                print(f"⚠️ Failed to load base64 file: {e}")
        
        # Method 2: Load from PNG file and encode
        image_file = '/Users/ashish_kumar/aiml/test_image.png'
        if os.path.exists(image_file):
            try:
                with open(image_file, 'rb') as f:
                    image_data = f.read()
                test_data['from_png_file'] = base64.b64encode(image_data).decode('utf-8')
                print(f"✅ Loaded and encoded PNG: {len(test_data['from_png_file'])} characters")
            except Exception as e:
                print(f"⚠️ Failed to load PNG file: {e}")
        
        # Method 3: Create simple test image
        try:
            from PIL import Image, ImageDraw
            import io
            
            # Create a simple white image with black text
            img = Image.new('RGB', (400, 100), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 30), 'HELLO WORLD TEST OCR', fill='black')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            test_data['generated_simple'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            print(f"✅ Generated simple test image: {len(test_data['generated_simple'])} characters")
            
        except Exception as e:
            print(f"⚠️ Failed to generate test image: {e}")
        
        return test_data
    
    def test_endpoint_inference(self, image_base64, test_name="Test", method='tesseract'):
        """Test the endpoint with given image data"""
        print(f"\n🧪 Running {test_name}...")
        print(f"📊 Image size: {len(image_base64)} characters")
        print(f"🔧 Method: {method}")
        
        # Prepare payload
        payload = {
            'image': image_base64,
            'method': method
        }
        
        try:
            start_time = time.time()
            
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Accept='application/json',
                Body=json.dumps(payload)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"⏱️ Inference time: {duration:.2f} seconds")
            
            # Parse response
            response_body = response['Body'].read().decode('utf-8')
            print(f"📤 Response length: {len(response_body)} characters")
            
            try:
                result = json.loads(response_body)
                print("✅ Response successfully parsed as JSON")
                print(f"🔑 Response keys: {list(result.keys())}")
                
                # Extract and display text
                extracted_text = None
                if 'extracted_text' in result:
                    extracted_text = result['extracted_text']
                elif 'text' in result:
                    extracted_text = result['text']
                elif 'result' in result:
                    extracted_text = result['result']
                
                if extracted_text:
                    print(f"📝 Extracted text: '{extracted_text.strip()}'")
                else:
                    print(f"📄 Full response: {json.dumps(result, indent=2)}")
                
                return True, result
                
            except json.JSONDecodeError:
                print(f"⚠️ Response is not valid JSON: {response_body[:200]}...")
                return False, response_body
                
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            print(f"💥 Error type: {type(e).__name__}")
            
            # Provide specific error guidance
            error_str = str(e).lower()
            if 'timeout' in error_str or 'time' in error_str:
                print("💡 Tip: This appears to be a timeout. The endpoint might be cold starting.")
            elif 'throttling' in error_str:
                print("💡 Tip: Request is being throttled. Wait a moment and try again.")
            elif 'model' in error_str:
                print("💡 Tip: There might be an issue with the model or inference code.")
            
            return False, str(e)
    
    def run_comprehensive_tests(self):
        """Run all test scenarios"""
        print("🚀 Starting Comprehensive SageMaker Endpoint Tests")
        print("=" * 70)
        print(f"⏰ Test started at: {datetime.now()}")
        
        # Step 1: Get endpoint name
        if not self.get_endpoint_name():
            print("❌ Could not determine endpoint name. Exiting.")
            return False
        
        # Step 2: Check endpoint status
        if not self.check_endpoint_status():
            print("❌ Endpoint is not ready for inference. Exiting.")
            return False
        
        # Step 3: Load test data
        test_data = self.load_test_data()
        if not test_data:
            print("❌ No test data available. Exiting.")
            return False
        
        # Step 4: Run inference tests
        print("\n" + "=" * 50)
        print("🧪 RUNNING INFERENCE TESTS")
        print("=" * 50)
        
        test_results = []
        
        # Test with different data sources and methods
        test_scenarios = []
        
        # Add available test data with different methods
        for data_name, image_data in test_data.items():
            test_scenarios.extend([
                (f"{data_name} - Tesseract", image_data, 'tesseract'),
                (f"{data_name} - PyTorch", image_data, 'pytorch'),
                (f"{data_name} - Both methods", image_data, 'both')
            ])
        
        # Run each test scenario
        for test_name, image_data, method in test_scenarios[:6]:  # Limit to 6 tests
            success, result = self.test_endpoint_inference(image_data, test_name, method)
            test_results.append((test_name, success, result))
        
        # Step 5: Summary
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        successful_tests = sum(1 for _, success, _ in test_results if success)
        total_tests = len(test_results)
        
        for test_name, success, result in test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n📈 Overall Results: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Your SageMaker endpoint is working correctly.")
            return True
        elif successful_tests > 0:
            print("⚠️ Some tests passed. Check the failed tests for issues.")
            return False
        else:
            print("💥 All tests failed. Please check your endpoint configuration.")
            return False

def main():
    """Main function to run the comprehensive tests"""
    try:
        tester = SageMakerEndpointTester()
        success = tester.run_comprehensive_tests()
        
        if success:
            print("\n🏆 Comprehensive testing completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed. Please review the results above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
