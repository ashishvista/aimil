#!/bin/bash
# test_runner.sh - Smart test runner for SageMaker endpoint testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 SageMaker Endpoint Test Runner${NC}"
echo "=================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check AWS profile
check_aws_profile() {
    echo -e "${YELLOW}🔍 Checking AWS configuration...${NC}"
    
    if ! command_exists aws; then
        echo -e "${RED}❌ AWS CLI not found. Please install AWS CLI.${NC}"
        exit 1
    fi
    
    if ! aws sts get-caller-identity --profile test-prod >/dev/null 2>&1; then
        echo -e "${RED}❌ AWS profile 'test-prod' not configured or invalid.${NC}"
        echo "Please configure your AWS profile with: aws configure --profile test-prod"
        exit 1
    fi
    
    echo -e "${GREEN}✅ AWS profile 'test-prod' is valid${NC}"
}

# Function to check Python dependencies
check_python_deps() {
    echo -e "${YELLOW}🔍 Checking Python dependencies...${NC}"
    
    if ! command_exists python3; then
        echo -e "${RED}❌ Python 3 not found.${NC}"
        exit 1
    fi
    
    # Check if required packages are installed
    python3 -c "import boto3, PIL" 2>/dev/null || {
        echo -e "${YELLOW}⚠️ Missing Python dependencies. Installing...${NC}"
        
        if [ -d "test_venv" ]; then
            echo -e "${BLUE}🔄 Using existing virtual environment...${NC}"
            source test_venv/bin/activate
        else
            echo -e "${BLUE}🔄 Creating virtual environment...${NC}"
            python3 -m venv test_venv
            source test_venv/bin/activate
        fi
        
        pip install boto3 pillow >/dev/null 2>&1 || {
            echo -e "${RED}❌ Failed to install Python dependencies.${NC}"
            exit 1
        }
    }
    
    echo -e "${GREEN}✅ Python dependencies are ready${NC}"
}

# Function to check test data
check_test_data() {
    echo -e "${YELLOW}🔍 Checking test data...${NC}"
    
    if [ ! -f "test_image_b64.txt" ] || [ ! -s "test_image_b64.txt" ]; then
        echo -e "${YELLOW}⚠️ test_image_b64.txt is missing or empty. Creating it...${NC}"
        
        if [ -f "test_image.png" ]; then
            python3 -c "
import base64
with open('test_image.png', 'rb') as f:
    image_data = f.read()
base64_string = base64.b64encode(image_data).decode('utf-8')
with open('test_image_b64.txt', 'w') as f:
    f.write(base64_string)
print('✅ Created test_image_b64.txt')
"
        else
            echo -e "${RED}❌ No test image found (test_image.png or test_image_b64.txt)${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}✅ Test data is ready${NC}"
}

# Function to check SageMaker endpoint
check_endpoint() {
    echo -e "${YELLOW}🔍 Checking SageMaker endpoint...${NC}"
    
    if ! command_exists terraform; then
        echo -e "${YELLOW}⚠️ Terraform not found. Will try to find endpoint manually.${NC}"
        return 0
    fi
    
    ENDPOINT_NAME=$(terraform output -raw sagemaker_endpoint_name 2>/dev/null || echo "")
    
    if [ -z "$ENDPOINT_NAME" ]; then
        echo -e "${YELLOW}⚠️ Could not get endpoint name from Terraform.${NC}"
        return 0
    fi
    
    echo -e "${GREEN}✅ Found endpoint: $ENDPOINT_NAME${NC}"
    
    # Check if endpoint is InService
    ENDPOINT_STATUS=$(aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --profile test-prod --region ap-south-1 --query 'EndpointStatus' --output text 2>/dev/null || echo "UNKNOWN")
    
    if [ "$ENDPOINT_STATUS" = "InService" ]; then
        echo -e "${GREEN}✅ Endpoint is InService and ready for testing${NC}"
    else
        echo -e "${YELLOW}⚠️ Endpoint status: $ENDPOINT_STATUS${NC}"
        echo -e "${YELLOW}   The endpoint might not be ready for testing yet.${NC}"
    fi
}

# Function to run the appropriate test
run_test() {
    echo ""
    echo -e "${BLUE}🧪 Which test would you like to run?${NC}"
    echo "1) Quick Test (fast basic validation)"
    echo "2) Comprehensive Test (thorough testing with multiple scenarios)"
    echo "3) Exit"
    echo ""
    
    while true; do
        read -p "Enter your choice (1-3): " choice
        case $choice in
            1)
                echo -e "${BLUE}🔄 Running Quick Test...${NC}"
                python3 test_sagemaker_quick.py
                break
                ;;
            2)
                echo -e "${BLUE}🔄 Running Comprehensive Test...${NC}"
                python3 test_sagemaker_comprehensive.py
                break
                ;;
            3)
                echo -e "${YELLOW}👋 Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Invalid choice. Please enter 1, 2, or 3.${NC}"
                ;;
        esac
    done
}

# Main execution
main() {
    # Run all checks
    check_aws_profile
    check_python_deps
    check_test_data
    check_endpoint
    
    echo ""
    echo -e "${GREEN}✅ All prerequisites are ready!${NC}"
    echo ""
    
    # Activate virtual environment if it exists
    if [ -d "test_venv" ]; then
        source test_venv/bin/activate
        echo -e "${BLUE}🔄 Activated virtual environment${NC}"
    fi
    
    # Run the test
    run_test
}

# Check if we're in the right directory
if [ ! -f "main.tf" ] || [ ! -f "sagemaker_endpoint.tf" ]; then
    echo -e "${RED}❌ Please run this script from the project root directory (where main.tf is located)${NC}"
    exit 1
fi

main "$@"
