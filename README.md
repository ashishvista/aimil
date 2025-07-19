# OCR SageMaker Pipeline with Terraform

This project creates a complete OCR (Optical Character Recognition) pipeline using AWS SageMaker, Lambda, and API Gateway, all provisioned with Terraform.

## Architecture Overview

The solution includes:

1. **SageMaker Pipeline**: For training and deploying OCR models
2. **SageMaker Endpoint**: For real-time OCR inference
3. **API Gateway**: RESTful API for file uploads and OCR processing
4. **Lambda Functions**: 
   - Presigned URL generation for secure file uploads
   - OCR processing using SageMaker endpoint
5. **S3 Buckets**: For storing training data, models, and uploaded files

## Features

- **Dual OCR Methods**: 
  - Traditional Tesseract OCR for immediate results
  - Custom PyTorch model for enhanced accuracy
- **Secure File Upload**: Presigned URLs for direct S3 uploads
- **JSON Response**: Structured output with text extraction and bounding boxes
- **CORS Enabled**: Ready for web application integration
- **Scalable**: Auto-scaling SageMaker endpoints

## Prerequisites

- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- Docker (for custom container images, if needed)

## Quick Start

### 1. Clone and Setup

```bash
cd /Users/ashish_kumar/aiml
```

### 2. Configure Variables

Edit `terraform.tfvars` to customize your deployment:

```hcl
aws_region = "us-east-1"
project_name = "my-ocr-pipeline"
endpoint_instance_type = "ml.m5.large"
training_instance_type = "ml.m5.xlarge"
```

### 3. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply configuration
terraform apply
```

### 4. Get API Endpoints

After deployment, get your API Gateway URL:

```bash
terraform output api_gateway_url
```

## API Usage

### Upload File

1. **Get Presigned URL**:
```bash
curl -X POST https://your-api-gateway-url/prod/upload \
  -H "Content-Type: application/json" \
  -d '{
    "file_extension": "jpg",
    "content_type": "image/jpeg"
  }'
```

2. **Upload File**:
```bash
curl -X PUT "presigned-upload-url" \
  -H "Content-Type: image/jpeg" \
  --data-binary @your-image.jpg
```

### Process OCR

```bash
curl -X POST https://your-api-gateway-url/prod/process \
  -H "Content-Type: application/json" \
  -d '{
    "s3_key": "uploads/your-file-key.jpg",
    "method": "both"
  }'
```

**Response Example**:
```json
{
  "tesseract": {
    "full_text": "Extracted text from image",
    "word_boxes": [
      {
        "text": "Extracted",
        "confidence": 0.95,
        "bbox": {
          "left": 10,
          "top": 20,
          "width": 80,
          "height": 25
        }
      }
    ]
  },
  "pytorch": {
    "predicted_text": "Custom model prediction",
    "confidence": 0.87
  },
  "image_info": {
    "width": 800,
    "height": 600,
    "mode": "RGB"
  }
}
```

## Pipeline Management

### Trigger Training Pipeline

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name ocr-pipeline-pipeline \
  --region us-east-1
```

### Monitor Pipeline Status

```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name ocr-pipeline-pipeline \
  --region us-east-1
```

## Customization

### Custom OCR Model

To use your own OCR model:

1. Update `scripts/train.py` with your training logic
2. Modify `scripts/inference.py` for your model's inference
3. Update the Docker image URI in `variables.tf`

### Custom Preprocessing

Modify `scripts/preprocess.py` to handle your specific data format and preprocessing requirements.

## Monitoring and Logging

- **CloudWatch Logs**: All Lambda functions log to CloudWatch
- **SageMaker Logs**: Training and endpoint logs available in CloudWatch
- **API Gateway Logs**: Request/response logging enabled

## Security

- IAM roles with minimal required permissions
- S3 buckets with encryption enabled
- API Gateway with CORS configured
- Presigned URLs for secure file uploads

## Cost Optimization

- SageMaker endpoints can be configured for auto-scaling
- Consider using Spot instances for training jobs
- Set up CloudWatch alarms for cost monitoring

## Troubleshooting

### Common Issues

1. **Endpoint Creation Fails**: Check IAM permissions and model artifacts
2. **Lambda Timeout**: Increase timeout in `lambda.tf`
3. **OCR Quality**: Ensure images are high resolution and well-lit

### Debug Commands

```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name your-endpoint-name

# View Lambda logs
aws logs tail /aws/lambda/ocr-pipeline-ocr-processor

# Test endpoint directly
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name your-endpoint-name \
  --content-type application/json \
  --body '{"image": "base64-encoded-image"}' \
  output.json
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

## File Structure

```
.
├── main.tf                 # Main Terraform configuration
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── iam.tf                  # IAM roles and policies
├── sagemaker_pipeline.tf   # SageMaker pipeline definition
├── sagemaker_endpoint.tf   # SageMaker model and endpoint
├── lambda.tf              # Lambda functions
├── api_gateway.tf         # API Gateway configuration
├── scripts/
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   ├── preprocess.py      # Data preprocessing
│   └── requirements.txt   # Python dependencies
└── lambda/
    ├── lambda_function.py # OCR processing Lambda
    └── presigned_url.py   # Upload URL generation Lambda
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

pipeline_name = "ocr-pipeline-pipeline"
sagemaker_bucket_name = "ocr-pipeline-sagemaker-sv1mmdug"
sagemaker_endpoint_name = "ocr-pipeline-endpoint-sv1mmdug"
upload_bucket_name = "ocr-pipeline-uploads-sv1mmdug"