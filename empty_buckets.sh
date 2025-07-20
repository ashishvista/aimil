#!/bin/bash

# Empty S3 buckets before Terraform destroy
# This script removes all objects and versions from S3 buckets to allow clean destruction

set -e

echo "🧹 Emptying S3 buckets for clean Terraform destroy..."

# AWS Profile configuration
AWS_PROFILE="test-prod"

# Get bucket names from Terraform outputs
UPLOAD_BUCKET=$(terraform output -raw upload_bucket_name 2>/dev/null || echo "")
SAGEMAKER_BUCKET=$(terraform output -raw sagemaker_bucket_name 2>/dev/null || echo "")

# Function to empty a bucket
empty_bucket() {
    local bucket_name=$1
    
    if [ -z "$bucket_name" ]; then
        echo "⚠️  Bucket name is empty, skipping..."
        return
    fi
    
    echo "🗑️  Emptying bucket: $bucket_name"
    
    # Check if bucket exists
    if aws s3api head-bucket --bucket "$bucket_name" --profile $AWS_PROFILE 2>/dev/null; then
        echo "📦 Found bucket: $bucket_name"
        
        # Remove all objects (including versions if versioning is enabled)
        echo "🧹 Removing all objects and versions..."
        aws s3api delete-objects \
            --bucket "$bucket_name" \
            --profile $AWS_PROFILE \
            --delete "$(aws s3api list-object-versions \
                --bucket "$bucket_name" \
                --profile $AWS_PROFILE \
                --output json \
                --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}')" \
            2>/dev/null || echo "No versions to delete"
        
        # Remove delete markers
        echo "🗑️  Removing delete markers..."
        aws s3api delete-objects \
            --bucket "$bucket_name" \
            --profile $AWS_PROFILE \
            --delete "$(aws s3api list-object-versions \
                --bucket "$bucket_name" \
                --profile $AWS_PROFILE \
                --output json \
                --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}')" \
            2>/dev/null || echo "No delete markers to remove"
        
        # Alternative method using s3 rm (for current versions)
        echo "🧽 Using s3 rm for final cleanup..."
        aws s3 rm "s3://$bucket_name" --recursive --profile $AWS_PROFILE 2>/dev/null || echo "No objects to remove"
        
        echo "✅ Bucket $bucket_name emptied successfully"
    else
        echo "⚠️  Bucket $bucket_name does not exist or is not accessible"
    fi
}

# Check AWS credentials
if ! aws sts get-caller-identity --profile $AWS_PROFILE >/dev/null 2>&1; then
    echo "❌ AWS credentials not configured or expired"
    echo "💡 Please run: aws configure or refresh your credentials"
    exit 1
fi

echo "🔍 Current AWS identity:"
aws sts get-caller-identity --profile $AWS_PROFILE

# Empty both buckets
if [ -n "$UPLOAD_BUCKET" ]; then
    empty_bucket "$UPLOAD_BUCKET"
else
    echo "⚠️  Upload bucket name not found in Terraform outputs"
fi

if [ -n "$SAGEMAKER_BUCKET" ]; then
    empty_bucket "$SAGEMAKER_BUCKET"
else
    echo "⚠️  SageMaker bucket name not found in Terraform outputs"
fi

# Also try to empty buckets by pattern if outputs are not available
echo "🔍 Searching for OCR pipeline buckets by pattern..."
BUCKETS=$(aws s3 ls --profile $AWS_PROFILE | grep "ocr-pipeline" | awk '{print $3}' || echo "")

if [ -n "$BUCKETS" ]; then
    echo "📋 Found buckets matching pattern:"
    echo "$BUCKETS"
    
    for bucket in $BUCKETS; do
        empty_bucket "$bucket"
    done
else
    echo "ℹ️  No buckets found matching 'ocr-pipeline' pattern"
fi

echo "🎉 S3 bucket cleanup complete!"
echo "💡 You can now run 'terraform destroy' safely"
