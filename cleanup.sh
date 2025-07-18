#!/bin/bash

# Cleanup script for OCR Pipeline
# This script destroys all AWS resources created by Terraform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to empty S3 buckets before destruction
empty_s3_buckets() {
    print_status "Emptying S3 buckets..."
    
    # Get bucket names from Terraform output
    local sagemaker_bucket=$(terraform output -raw sagemaker_bucket_name 2>/dev/null || echo "")
    local upload_bucket=$(terraform output -raw upload_bucket_name 2>/dev/null || echo "")
    
    if [ -n "$sagemaker_bucket" ]; then
        print_status "Emptying SageMaker bucket: $sagemaker_bucket"
        aws s3 rm s3://$sagemaker_bucket --recursive 2>/dev/null || true
    fi
    
    if [ -n "$upload_bucket" ]; then
        print_status "Emptying upload bucket: $upload_bucket"
        aws s3 rm s3://$upload_bucket --recursive 2>/dev/null || true
    fi
}

# Function to stop any running SageMaker endpoints
stop_sagemaker_endpoints() {
    print_status "Checking for running SageMaker endpoints..."
    
    local endpoint_name=$(terraform output -raw sagemaker_endpoint_name 2>/dev/null || echo "")
    
    if [ -n "$endpoint_name" ]; then
        print_status "Deleting SageMaker endpoint: $endpoint_name"
        aws sagemaker delete-endpoint --endpoint-name "$endpoint_name" 2>/dev/null || true
        
        # Wait for endpoint deletion
        print_status "Waiting for endpoint deletion to complete..."
        aws sagemaker wait endpoint-deleted --endpoint-name "$endpoint_name" 2>/dev/null || true
    fi
}

# Main cleanup function
main() {
    echo "OCR Pipeline Cleanup Script"
    echo "=========================="
    
    # Warning message
    print_warning "This will PERMANENTLY DELETE all AWS resources created by this Terraform project."
    print_warning "This action cannot be undone!"
    echo ""
    
    # Check if we should skip confirmation
    if [[ "$1" != "--force" ]]; then
        read -p "Are you sure you want to continue? Type 'yes' to confirm: " -r
        echo ""
        if [[ ! $REPLY == "yes" ]]; then
            print_status "Cleanup cancelled."
            exit 0
        fi
    fi
    
    # Check if Terraform is initialized
    if [ ! -d ".terraform" ]; then
        print_error "Terraform not initialized. Run 'terraform init' first."
        exit 1
    fi
    
    # Empty S3 buckets first (required before deletion)
    empty_s3_buckets
    
    # Stop SageMaker endpoints
    stop_sagemaker_endpoints
    
    # Run terraform destroy
    print_status "Running Terraform destroy..."
    terraform destroy -auto-approve
    
    # Clean up local files
    print_status "Cleaning up local files..."
    rm -f terraform.tfstate*
    rm -f tfplan
    rm -f deployment_info.txt
    rm -f test_image.jpg
    rm -f lambda_function.zip
    rm -f presigned_url.zip
    
    print_status "🗑️  Cleanup completed successfully!"
    print_status "All AWS resources have been destroyed and local files cleaned up."
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--force|--help]"
        echo ""
        echo "Options:"
        echo "  --force    Skip confirmation prompt"
        echo "  --help     Show this help"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
