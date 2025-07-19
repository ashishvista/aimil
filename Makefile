# OCR Pipeline Makefile
# This Makefile provides convenient commands for managing the OCR pipeline

.PHONY: help init plan deploy test cleanup demo clean install-deps check-deps

# Default target
help:
	@echo "OCR Pipeline Management Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install-deps  - Install required dependencies"
	@echo "  make check-deps    - Check if dependencies are installed"
	@echo "  make init          - Initialize Terraform"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  make plan          - Show Terraform plan"
	@echo "  make deploy        - Deploy the OCR pipeline"
	@echo "  make test          - Test the deployed pipeline"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make demo          - Open demo web interface"
	@echo "  make cleanup       - Destroy all AWS resources"
	@echo "  make clean         - Clean local files"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make all           - Full deployment (init + deploy + test)"
	@echo "  make quick-deploy  - Deploy without confirmation"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@command -v terraform >/dev/null 2>&1 || { echo "❌ Terraform not installed"; exit 1; }
	@command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI not installed"; exit 1; }
	@aws sts get-caller-identity --profile=test-prod >/dev/null 2>&1 || { echo "❌ AWS credentials not configured"; exit 1; }
	@echo "✅ All dependencies are installed and configured"

# Install dependencies (macOS)
install-deps:
	@echo "Installing dependencies for macOS..."
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "❌ Homebrew not installed. Please install it first: https://brew.sh/"; \
		exit 1; \
	fi
	@if ! command -v terraform >/dev/null 2>&1; then \
		echo "📦 Installing Terraform..."; \
		brew install terraform; \
	fi
	@if ! command -v aws >/dev/null 2>&1; then \
		echo "📦 Installing AWS CLI..."; \
		brew install awscli; \
	fi
	@if ! command -v jq >/dev/null 2>&1; then \
		echo "📦 Installing jq..."; \
		brew install jq; \
	fi
	@echo "✅ Dependencies installed successfully"
	@echo "⚠️  Please configure AWS credentials: aws configure"

# Initialize Terraform
init: check-deps
	@echo "🚀 Initializing Terraform..."
	terraform init

# Show Terraform plan
plan: init
	@echo "📋 Generating Terraform plan..."
	terraform plan

# Deploy infrastructure
deploy: check-deps
	@echo "🚀 Deploying OCR pipeline..."
	./deploy.sh

# Quick deploy without confirmation
quick-deploy: check-deps
	@echo "🚀 Quick deploying OCR pipeline..."
	./deploy.sh --auto

# Test the deployed pipeline
test:
	@echo "🧪 Testing OCR pipeline..."
	./test_pipeline.sh

# Open demo web interface
demo:
	@echo "🌐 Opening demo web interface..."
	@if command -v open >/dev/null 2>&1; then \
		open demo.html; \
	else \
		echo "Please open demo.html in your web browser"; \
	fi

# Cleanup AWS resources
cleanup:
	@echo "🗑️  Cleaning up AWS resources..."
	./cleanup.sh

# Clean local files
clean:
	@echo "🧹 Cleaning local files..."
	rm -f terraform.tfstate*
	rm -f tfplan
	rm -f deployment_info.txt
	rm -f test_image.jpg
	rm -f lambda_function.zip
	rm -f presigned_url.zip
	rm -rf .terraform/
	@echo "✅ Local files cleaned"

# Full deployment pipeline
all: check-deps init deploy test
	@echo "🎉 Full deployment completed successfully!"

# Show current status
status:
	@echo "📊 OCR Pipeline Status"
	@echo "====================="
	@if [ -f ".terraform/terraform.tfstate" ]; then \
		echo "Terraform: ✅ Initialized"; \
	else \
		echo "Terraform: ❌ Not initialized"; \
	fi
	@if [ -f "deployment_info.txt" ]; then \
		echo "Deployment: ✅ Active"; \
		echo ""; \
		cat deployment_info.txt; \
	else \
		echo "Deployment: ❌ Not deployed"; \
	fi

# Show logs from Lambda functions
logs:
	@echo "📋 Showing recent Lambda logs..."
	@if [ -f "deployment_info.txt" ]; then \
		echo "OCR Processor logs:"; \
		aws logs tail /aws/lambda/ocr-pipeline-ocr-processor --since 1h 2>/dev/null || echo "No logs found"; \
		echo ""; \
		echo "Presigned URL Generator logs:"; \
		aws logs tail /aws/lambda/ocr-pipeline-presigned-url --since 1h 2>/dev/null || echo "No logs found"; \
	else \
		echo "❌ Pipeline not deployed"; \
	fi

# Validate Terraform configuration
validate:
	@echo "✅ Validating Terraform configuration..."
	terraform validate
	terraform fmt -check=true
	@echo "✅ Configuration is valid"

# Format Terraform files
format:
	@echo "🎨 Formatting Terraform files..."
	terraform fmt -recursive
	@echo "✅ Files formatted"

# Show resource costs estimate (requires infracost)
costs:
	@if command -v infracost >/dev/null 2>&1; then \
		echo "💰 Estimating costs..."; \
		infracost breakdown --path .; \
	else \
		echo "❌ infracost not installed. Install it to see cost estimates."; \
		echo "   brew install infracost"; \
	fi
