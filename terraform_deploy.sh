#!/bin/bash

# Pure Terraform Deployment Script
# This script uses only Terraform to handle model creation and deployment

set -e

echo "🚀 Starting Pure Terraform OCR Pipeline Deployment..."

# AWS Profile configuration
AWS_PROFILE="test-prod"

# Check AWS credentials
if ! aws sts get-caller-identity --profile $AWS_PROFILE >/dev/null 2>&1; then
    echo "❌ AWS credentials not configured or expired"
    echo "💡 Please run: aws sso login --profile $AWS_PROFILE"
    exit 1
fi

echo "🔍 Current AWS identity:"
aws sts get-caller-identity --profile $AWS_PROFILE

# Step 1: Initialize Terraform if needed
if [ ! -d ".terraform" ]; then
    echo "🔧 Initializing Terraform..."
    terraform init
fi

# Step 2: Plan the deployment
echo "📋 Planning Terraform deployment..."
terraform plan -out=tfplan

echo ""
read -p "🤔 Do you want to proceed with the deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    rm -f tfplan
    exit 0
fi

# Step 3: Apply in stages for better control
echo "🏗️  Applying Terraform configuration in stages..."

# Stage 1: Core infrastructure (S3, IAM)
echo "📦 Stage 1: Creating core infrastructure..."
terraform apply -target=aws_s3_bucket.sagemaker_bucket \
                -target=aws_s3_bucket.upload_bucket \
                -target=aws_iam_role.sagemaker_execution_role \
                -target=aws_iam_role.lambda_execution_role \
                -target=aws_iam_role.sagemaker_pipeline_role \
                -target=random_string.suffix \
                -auto-approve

# Stage 2: Model creation and upload
echo "🧠 Stage 2: Creating and uploading model artifact..."
terraform apply -target=null_resource.create_model_artifact \
                -target=aws_s3_object.model_artifact \
                -target=aws_s3_object.training_script \
                -target=aws_s3_object.inference_script \
                -target=aws_s3_object.requirements \
                -auto-approve

# Stage 3: SageMaker resources
echo "🤖 Stage 3: Creating SageMaker resources..."
terraform apply -target=aws_sagemaker_pipeline.ocr_pipeline \
                -target=aws_sagemaker_model.ocr_model \
                -target=null_resource.verify_model_creation \
                -auto-approve

# Stage 4: Lambda and API Gateway
echo "⚡ Stage 4: Creating Lambda functions and API Gateway..."
terraform apply -target=null_resource.build_lambda_layer \
                -target=aws_lambda_layer_version.dependencies \
                -target=aws_lambda_function.ocr_processor \
                -target=aws_lambda_function.presigned_url_generator \
                -target=aws_api_gateway_rest_api.ocr_api \
                -auto-approve

# Stage 5: Final deployment (endpoints and remaining resources)
echo "🌐 Stage 5: Final deployment..."
terraform apply -auto-approve

# Cleanup
rm -f tfplan

# Get outputs
echo ""
echo "🎉 Deployment Complete!"
echo "========================"

if terraform output api_gateway_url >/dev/null 2>&1; then
    API_URL=$(terraform output -raw api_gateway_url)
    echo "API Gateway URL: $API_URL"
fi

if terraform output sagemaker_endpoint_name >/dev/null 2>&1; then
    ENDPOINT_NAME=$(terraform output -raw sagemaker_endpoint_name)
    echo "SageMaker Endpoint: $ENDPOINT_NAME"
fi

if terraform output upload_bucket_name >/dev/null 2>&1; then
    UPLOAD_BUCKET=$(terraform output -raw upload_bucket_name)
    echo "Upload Bucket: $UPLOAD_BUCKET"
fi

if terraform output pipeline_name >/dev/null 2>&1; then
    PIPELINE_NAME=$(terraform output -raw pipeline_name)
    echo "Pipeline Name: $PIPELINE_NAME"
fi

echo ""
echo "✅ All resources created using pure Terraform approach!"
echo "🔗 API Endpoints:"
echo "   - Upload: $API_URL/upload"
echo "   - Process: $API_URL/process"
echo ""
echo "📊 Model artifact managed by Terraform:"
echo "   - Created via null_resource.create_model_artifact"
echo "   - Uploaded via aws_s3_object.model_artifact"
echo "   - Verified via null_resource.verify_model_creation"
