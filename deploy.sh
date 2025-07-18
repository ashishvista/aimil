#!/bin/bash

# OCR Pipeline Deployment Script
# This script automates the deployment of the OCR SageMaker pipeline

set -e

echo "🚀 Starting OCR Pipeline Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    print_status "Prerequisites check passed ✓"
}

# Initialize Terraform
init_terraform() {
    print_status "Initializing Terraform..."
    terraform init
    print_status "Terraform initialization complete ✓"
}

# Plan Terraform deployment
plan_terraform() {
    print_status "Planning Terraform deployment..."
    terraform plan -out=tfplan
    print_status "Terraform plan complete ✓"
}

# Apply Terraform configuration
apply_terraform() {
    print_status "Applying Terraform configuration..."
    terraform apply tfplan
    print_status "Terraform apply complete ✓"
}

# Get outputs
get_outputs() {
    print_status "Getting deployment outputs..."
    
    API_URL=$(terraform output -raw api_gateway_url)
    ENDPOINT_NAME=$(terraform output -raw sagemaker_endpoint_name)
    UPLOAD_BUCKET=$(terraform output -raw upload_bucket_name)
    PIPELINE_NAME=$(terraform output -raw pipeline_name)
    
    echo ""
    echo "🎉 Deployment Complete!"
    echo "========================"
    echo "API Gateway URL: $API_URL"
    echo "SageMaker Endpoint: $ENDPOINT_NAME"
    echo "Upload Bucket: $UPLOAD_BUCKET"
    echo "Pipeline Name: $PIPELINE_NAME"
    echo ""
    
    # Save outputs to file
    cat > deployment_info.txt << EOF
OCR Pipeline Deployment Information
==================================

API Gateway URL: $API_URL
SageMaker Endpoint: $ENDPOINT_NAME
Upload Bucket: $UPLOAD_BUCKET
Pipeline Name: $PIPELINE_NAME

API Endpoints:
- Upload URL: $API_URL/prod/upload
- Process URL: $API_URL/prod/process

Usage Examples:
1. Get presigned URL:
   curl -X POST $API_URL/prod/upload -H "Content-Type: application/json" -d '{"file_extension": "jpg", "content_type": "image/jpeg"}'

2. Process OCR:
   curl -X POST $API_URL/prod/process -H "Content-Type: application/json" -d '{"s3_key": "your-file-key", "method": "tesseract"}'
EOF
    
    print_status "Deployment information saved to deployment_info.txt ✓"
}

# Test the deployment
test_deployment() {
    print_status "Testing deployment..."
    
    # Test presigned URL generation
    print_status "Testing presigned URL generation..."
    curl -s -X POST "$API_URL/prod/upload" \
        -H "Content-Type: application/json" \
        -d '{"file_extension": "jpg", "content_type": "image/jpeg"}' | jq .
    
    print_status "Basic deployment test complete ✓"
}

# Main execution
main() {
    echo "OCR Pipeline Deployment Script"
    echo "=============================="
    
    # Check if we should skip confirmation
    if [[ "$1" != "--auto" ]]; then
        echo ""
        print_warning "This script will deploy AWS resources that may incur costs."
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Deployment cancelled."
            exit 0
        fi
    fi
    
    check_prerequisites
    init_terraform
    plan_terraform
    
    if [[ "$1" != "--auto" ]]; then
        echo ""
        print_warning "Please review the Terraform plan above."
        read -p "Do you want to apply these changes? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Deployment cancelled."
            exit 0
        fi
    fi
    
    apply_terraform
    get_outputs
    
    # Optional test
    if command -v jq &> /dev/null && [[ "$1" != "--no-test" ]]; then
        test_deployment
    else
        print_warning "Skipping tests (jq not installed or --no-test flag used)"
    fi
    
    echo ""
    print_status "🎉 OCR Pipeline deployment completed successfully!"
    print_status "Check deployment_info.txt for API details."
}

# Run main function with all arguments
main "$@"
