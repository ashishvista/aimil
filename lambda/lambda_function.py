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
    print("🚀 Lambda handler started")
    print(f"📨 Event: {json.dumps(event, indent=2, default=str)}")
    print(f"⏱️  Context: {context}")
    
    try:
        print("🔄 Parsing request...")
        # Parse the request
        if 'body' in event:
            print("📦 Found 'body' in event")
            if isinstance(event['body'], str):
                print("🔄 Parsing JSON body...")
                body = json.loads(event['body'])
                print("✅ JSON body parsed successfully")
            else:
                print("📦 Body is already parsed")
                body = event['body']
        else:
            print("📦 Using event as body directly")
            body = event
            
        print(f"📋 Parsed body: {json.dumps(body, indent=2, default=str)}")
        
        print("🔄 Getting environment variables...")
        # Get environment variables
        endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']
        upload_bucket = os.environ['UPLOAD_BUCKET']
        
        print(f"🎯 Endpoint name: {endpoint_name}")
        print(f"🪣 Upload bucket: {upload_bucket}")
        
        logger.info(f"Processing OCR request with endpoint: {endpoint_name}")
        
        # Handle different input types
        if 's3_key' in body:
            # Process file from S3
            s3_key = body['s3_key']
            logger.info(f"Processing file from S3: {s3_key}")
            print(f"📁 S3 Key: {s3_key}")
            print(f"📁 Upload Bucket: {upload_bucket}")
            
            try:
                print("🔄 Starting S3 object download...")
                # Download image from S3
                response = s3_client.get_object(Bucket=upload_bucket, Key=s3_key)
                print("✅ S3 get_object successful")
                
                print("🔄 Reading image data from S3 response...")
                image_data = response['Body'].read()
                print(f"✅ Image data read successfully, size: {len(image_data)} bytes")
                
            except Exception as s3_error:
                print(f"❌ S3 Error: {s3_error}")
                logger.error(f"S3 Error: {s3_error}")
                raise
            
        elif 'image_base64' in body:
            # Process base64 encoded image
            print("🔄 Processing base64 encoded image...")
            try:
                image_data = base64.b64decode(body['image_base64'])
                print(f"✅ Base64 decode successful, size: {len(image_data)} bytes")
            except Exception as b64_error:
                print(f"❌ Base64 decode error: {b64_error}")
                logger.error(f"Base64 decode error: {b64_error}")
                raise
            
        else:
            print("❌ No valid image data provided")
            logger.error("No image data provided in request")
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
        
        print("🔄 Preparing SageMaker payload...")
        # Prepare the payload for SageMaker
        try:
            payload = {
                'image': base64.b64encode(image_data).decode('utf-8'),
                'method': body.get('method', 'tesseract')  # tesseract, pytorch, or both
            }
            print(f"✅ Payload prepared, method: {payload['method']}")
            print(f"📊 Payload image size: {len(payload['image'])} chars")
        except Exception as payload_error:
            print(f"❌ Payload preparation error: {payload_error}")
            logger.error(f"Payload preparation error: {payload_error}")
            raise
        
        print(f"🔄 Invoking SageMaker endpoint: {endpoint_name}")
        # Invoke SageMaker endpoint
        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            print("✅ SageMaker endpoint invoked successfully")
        except Exception as sagemaker_error:
            print(f"❌ SageMaker invocation error: {sagemaker_error}")
            logger.error(f"SageMaker invocation error: {sagemaker_error}")
            raise
        
        print("🔄 Parsing SageMaker response...")
        # Parse the response
        try:
            result = json.loads(response['Body'].read().decode())
            print("✅ SageMaker response parsed successfully")
            print(f"📋 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        except Exception as parse_error:
            print(f"❌ Response parsing error: {parse_error}")
            logger.error(f"Response parsing error: {parse_error}")
            raise
        
        # Add metadata
        print("🔄 Adding metadata to result...")
        result['metadata'] = {
            'endpoint_used': endpoint_name,
            'processing_method': body.get('method', 'tesseract'),
            'input_type': 's3_key' if 's3_key' in body else 'base64'
        }
        print("✅ Metadata added successfully")
        
        print("🎉 OCR processing completed successfully!")
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
        print(f"💥 CRITICAL ERROR: {str(e)}")
        print(f"💥 Error type: {type(e).__name__}")
        import traceback
        print(f"💥 Traceback: {traceback.format_exc()}")
        logger.error(f"Error processing OCR request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'error_type': type(e).__name__
            })
        }
