#!/bin/bash

# Pure Terraform Model Artifact Fix
# This script uses only Terraform to create and upload the model artifact

set -e

echo "🔧 Fixing missing SageMaker model artifact using pure Terraform..."

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

# Step 1: Create model artifact using null resource
echo "🏗️  Step 1: Creating model artifact via Terraform..."
terraform apply -target=null_resource.create_model_artifact -auto-approve

# Step 2: Upload model artifact to S3 using Terraform
echo "☁️  Step 2: Uploading model to S3 via Terraform..."
terraform apply -target=aws_s3_object.model_artifact -auto-approve

# Step 3: Create SageMaker model using Terraform
echo "🧠 Step 3: Creating SageMaker model via Terraform..."
terraform apply -target=aws_sagemaker_model.ocr_model -auto-approve

# Step 4: Verify model creation using null resource
echo "✅ Step 4: Verifying model creation via Terraform..."
terraform apply -target=null_resource.verify_model_creation -auto-approve

echo "🎉 Model artifact fix complete using pure Terraform!"
echo "📋 Summary of Terraform resources used:"
echo "   ✅ null_resource.create_model_artifact - Created model.tar.gz"
echo "   ✅ aws_s3_object.model_artifact - Uploaded to S3"
echo "   ✅ aws_sagemaker_model.ocr_model - Created SageMaker model"
echo "   ✅ null_resource.verify_model_creation - Verified deployment"
echo ""
echo "💡 You can now continue with: terraform apply -auto-approve"
