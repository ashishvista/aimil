import json
import boto3
import base64
import logging
from io import BytesIO
import os
import sys

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Debug: Print Python version and path info
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Debug: Check if PIL directory exists and list contents
try:
    import os
    pil_paths = [
        "/opt/python/lib/python3.9/site-packages/PIL",
        "/opt/python/PIL", 
        "/var/runtime/PIL"
    ]
    for path in pil_paths:
        if os.path.exists(path):
            print(f"✅ PIL directory found at: {path}")
            files = os.listdir(path)
            print(f"PIL directory contents (first 10): {files[:10]}")
            if "__init__.py" in files:
                print("✅ __init__.py found in PIL directory")
            imaging_files = [f for f in files if "imaging" in f.lower()]
            print(f"PIL imaging files: {imaging_files}")
        else:
            print(f"❌ PIL directory not found at: {path}")
            
    # Check site-packages directory
    site_packages = "/opt/python/lib/python3.9/site-packages"
    if os.path.exists(site_packages):
        print(f"✅ Site-packages directory exists: {site_packages}")
        packages = [d for d in os.listdir(site_packages) if os.path.isdir(os.path.join(site_packages, d))]
        print(f"Installed packages: {packages[:10]}")
    else:
        print(f"❌ Site-packages directory not found: {site_packages}")
        
except Exception as e:
    print(f"Debug error: {e}")

# Now try to import PIL
try:
    from PIL import Image
    print("✅ PIL imported successfully!")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")
    # Try alternative import paths
    try:
        sys.path.insert(0, '/opt/python/lib/python3.9/site-packages')
        from PIL import Image
        print("✅ PIL imported after adding to sys.path!")
    except ImportError as e2:
        print(f"❌ PIL import still failed after path modification: {e2}")
        raise

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to process OCR requests using SageMaker endpoint
    """
    try:
        # Parse the request
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event
        
        # Get environment variables
        endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']
        upload_bucket = os.environ['UPLOAD_BUCKET']
        
        logger.info(f"Processing OCR request with endpoint: {endpoint_name}")
        
        # Handle different input types
        if 's3_key' in body:
            # Process file from S3
            s3_key = body['s3_key']
            logger.info(f"Processing file from S3: {s3_key}")
            
            # Download image from S3
            response = s3_client.get_object(Bucket=upload_bucket, Key=s3_key)
            image_data = response['Body'].read()
            
        elif 'image_base64' in body:
            # Process base64 encoded image
            image_data = base64.b64decode(body['image_base64'])
            
        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS'
                },
                'body': json.dumps({'error': 'No image data provided. Use s3_key or image_base64.'})
            }
        
        # Prepare the payload for SageMaker
        payload = {
            'image': base64.b64encode(image_data).decode('utf-8'),
            'method': body.get('method', 'tesseract')  # tesseract, pytorch, or both
        }
        
        # Invoke SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        # Add metadata
        result['metadata'] = {
            'endpoint_used': endpoint_name,
            'processing_method': body.get('method', 'tesseract'),
            'input_type': 's3_key' if 's3_key' in body else 'base64'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps(result, indent=2)
        }
        
    except Exception as e:
        logger.error(f"Error processing OCR request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
