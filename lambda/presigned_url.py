import json
import boto3
import logging
import os
from datetime import datetime, timedelta
import uuid

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to generate presigned URLs for file upload
    """
    try:
        # Parse the request
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body']) if event['body'] else {}
            else:
                body = event['body']
        else:
            body = event
        
        # Get environment variables
        upload_bucket = os.environ['UPLOAD_BUCKET']
        
        # Get parameters from request
        file_extension = body.get('file_extension', 'jpg')
        content_type = body.get('content_type', 'image/jpeg')
        expires_in = body.get('expires_in', 3600)  # 1 hour default
        
        # Generate unique file key
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        file_key = f"uploads/{timestamp}_{unique_id}.{file_extension}"
        
        logger.info(f"Generating presigned URL for bucket: {upload_bucket}, key: {file_key}")
        
        # Generate presigned URL for PUT operation
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': upload_bucket,
                'Key': file_key,
                'ContentType': content_type
            },
            ExpiresIn=expires_in
        )
        
        # Generate presigned URL for GET operation (for verification)
        download_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': upload_bucket,
                'Key': file_key
            },
            ExpiresIn=expires_in
        )
        
        response_data = {
            'upload_url': presigned_url,
            'download_url': download_url,
            'file_key': file_key,
            'bucket': upload_bucket,
            'expires_in': expires_in,
            'content_type': content_type,
            'upload_instructions': {
                'method': 'PUT',
                'headers': {
                    'Content-Type': content_type
                },
                'note': 'Use the upload_url with PUT method to upload your file'
            }
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            },
            'body': json.dumps(response_data, indent=2)
        }
        
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
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
