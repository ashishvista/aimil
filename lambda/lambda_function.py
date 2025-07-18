import json
import boto3
import base64
import logging
from io import BytesIO
from PIL import Image
import os

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
