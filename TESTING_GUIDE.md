# SageMaker Endpoint Testing Guide

This guide explains how to test your deployed SageMaker OCR endpoint using the provided test scripts and data.

## 📁 Test Files Overview

### Test Scripts
- `test_runner.sh` - Smart test runner with prerequisite checks (recommended)
- `test_sagemaker_quick.py` - Quick endpoint validation test
- `test_sagemaker_comprehensive.py` - Thorough testing with multiple scenarios
- `test_sagemaker_direct.py` - Original direct endpoint test

### Test Data
- `test_image.png` - Original PNG test image (217KB)
- `test_image_b64.txt` - Base64 encoded version of the test image
- Generated test images created programmatically during testing

## 🚀 Quick Start

### Option 1: Using the Smart Test Runner (Recommended)
```bash
./test_runner.sh
```

The test runner will:
- ✅ Check AWS credentials and profile
- ✅ Verify Python dependencies
- ✅ Ensure test data is available
- ✅ Check endpoint status
- 🎯 Let you choose between quick or comprehensive testing

### Option 2: Direct Testing
```bash
# Quick test
python3 test_sagemaker_quick.py

# Comprehensive test
python3 test_sagemaker_comprehensive.py
```

## 📋 Prerequisites

### 1. AWS Configuration
- AWS CLI installed and configured
- Profile `test-prod` set up with proper credentials
- Permissions for SageMaker and related services

### 2. Python Dependencies
```bash
pip install boto3 pillow
```

### 3. Test Data
- The test runner will automatically create `test_image_b64.txt` if needed
- Or you can manually create it:
```bash
python3 -c "
import base64
with open('test_image.png', 'rb') as f:
    image_data = f.read()
base64_string = base64.b64encode(image_data).decode('utf-8')
with open('test_image_b64.txt', 'w') as f:
    f.write(base64_string)
print('Created test_image_b64.txt')
"
```

## 🧪 Test Types Explained

### Quick Test (`test_sagemaker_quick.py`)
- **Purpose**: Fast validation that the endpoint is working
- **Duration**: ~30 seconds
- **Tests**: Single inference with Tesseract method
- **Use when**: You want to quickly verify the endpoint is operational

### Comprehensive Test (`test_sagemaker_comprehensive.py`)
- **Purpose**: Thorough testing with multiple scenarios
- **Duration**: 2-5 minutes
- **Tests**: 
  - Multiple data sources (file, base64, generated)
  - Different OCR methods (Tesseract, PyTorch, both)
  - Error handling and edge cases
- **Use when**: You want to thoroughly validate all endpoint functionality

## 📊 Understanding Test Results

### Successful Test Output
```
✅ AWS session initialized with profile: test-prod, region: ap-south-1
✅ Got endpoint name from terraform: ocr-pipeline-endpoint-sv1mmdug
🔍 Endpoint Status: InService
⏱️ Inference time: 2.34 seconds
📝 Extracted text: 'HELLO WORLD TEST OCR'
🎉 Quick test PASSED!
```

### Common Error Patterns

#### Endpoint Not Ready
```
❌ Endpoint is not InService. Current status: Creating
```
**Solution**: Wait for endpoint to finish deploying (5-10 minutes)

#### Authentication Issues
```
❌ No credentials found for profile: test-prod
```
**Solution**: Configure AWS profile with `aws configure --profile test-prod`

#### Timeout Errors
```
❌ Inference failed: timeout
💡 Tip: This appears to be a timeout. The endpoint might be cold starting.
```
**Solution**: Cold start can take 30-60 seconds. Run test again.

## 🔧 Manual Testing

### Using AWS CLI
```bash
# Get endpoint name
ENDPOINT_NAME=$(terraform output -raw sagemaker_endpoint_name)

# Check endpoint status
aws sagemaker describe-endpoint \
  --endpoint-name "$ENDPOINT_NAME" \
  --profile test-prod \
  --region ap-south-1

# Test inference
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name "$ENDPOINT_NAME" \
  --content-type "application/json" \
  --body '{"image":"'"$(cat test_image_b64.txt)"'","method":"tesseract"}' \
  --profile test-prod \
  --region ap-south-1 \
  response.json

# View results
cat response.json
```

### Using Python Boto3
```python
import boto3
import json
import base64

# Initialize client
session = boto3.Session(profile_name='test-prod')
client = session.client('sagemaker-runtime', region_name='ap-south-1')

# Load test image
with open('test_image_b64.txt', 'r') as f:
    image_base64 = f.read().strip()

# Prepare payload
payload = {
    'image': image_base64,
    'method': 'tesseract'  # or 'pytorch' or 'both'
}

# Invoke endpoint
response = client.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse response
result = json.loads(response['Body'].read().decode())
print(f"OCR Result: {result.get('extracted_text', result)}")
```

## 🔍 Troubleshooting

### 1. Endpoint Issues
```bash
# Check endpoint logs
./view_endpoint_logs.sh

# Check endpoint status
./quick_endpoint_check.sh
```

### 2. Test Data Issues
- Ensure `test_image.png` exists and is readable
- Verify `test_image_b64.txt` is created and not empty
- Check image format is supported (PNG, JPEG)

### 3. Network/Permissions Issues
- Verify AWS profile has SageMaker permissions
- Check if endpoint is in the correct region
- Ensure security groups allow traffic (if using VPC)

### 4. Performance Issues
- First invocation (cold start) may take 30-60 seconds
- Subsequent calls should be faster (2-5 seconds)
- Large images will take longer to process

## 📈 Performance Expectations

### Typical Response Times
- **Cold Start**: 30-60 seconds (first request after idle)
- **Warm Requests**: 2-10 seconds (depending on image size)
- **Large Images (>1MB)**: 10-30 seconds

### Payload Size Limits
- **Maximum**: 6MB for SageMaker endpoints
- **Recommended**: <1MB for optimal performance
- **Test Image**: 289KB base64 (~217KB original)

## 🎯 Next Steps

After successful testing:

1. **Integration**: Use the same payload format in your application
2. **Monitoring**: Set up CloudWatch alerts for endpoint metrics
3. **Scaling**: Configure auto-scaling based on usage patterns
4. **Cost Optimization**: Consider endpoint instance types and scaling policies

## 📚 Related Files

- `lambda/lambda_function.py` - Lambda integration example
- `view_endpoint_logs.sh` - Check endpoint CloudWatch logs  
- `quick_endpoint_check.sh` - Quick endpoint status check
- `sagemaker_endpoint.tf` - Terraform endpoint configuration
