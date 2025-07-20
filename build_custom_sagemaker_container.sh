#!/bin/bash
# build_custom_sagemaker_container.sh
# Build and deploy custom SageMaker container with Tesseract OCR

set -e

# Configuration
AWS_PROFILE="test-prod"
AWS_REGION="ap-south-1"
AWS_ACCOUNT_ID="913197190703"
REPO_NAME="sagemaker-ocr-tesseract"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Building custom SageMaker container with Tesseract OCR${NC}"
echo "=================================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check AWS CLI and profile
if ! aws sts get-caller-identity --profile $AWS_PROFILE >/dev/null 2>&1; then
    echo -e "${RED}❌ AWS profile '$AWS_PROFILE' not configured or expired.${NC}"
    echo -e "${YELLOW}💡 Please run: aws sso login --profile $AWS_PROFILE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and AWS CLI are ready${NC}"

# Create ECR repository if it doesn't exist
echo -e "${BLUE}🏪 Creating ECR repository if needed...${NC}"
if ! aws ecr describe-repositories --repository-names $REPO_NAME --profile $AWS_PROFILE --region $AWS_REGION >/dev/null 2>&1; then
    echo -e "${YELLOW}📦 Creating new ECR repository: $REPO_NAME${NC}"
    aws ecr create-repository \
        --repository-name $REPO_NAME \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true
    echo -e "${GREEN}✅ ECR repository created${NC}"
else
    echo -e "${GREEN}✅ ECR repository already exists${NC}"
fi

# Get ECR login
echo -e "${BLUE}🔐 Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION --profile $AWS_PROFILE | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ ECR login successful${NC}"
else
    echo -e "${RED}❌ ECR login failed${NC}"
    exit 1
fi

# Build the Docker image
echo -e "${BLUE}🔨 Building Docker image...${NC}"
IMAGE_NAME="$REPO_NAME:$IMAGE_TAG"
docker build -f Dockerfile.ubuntu -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Tag image for ECR
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
echo -e "${BLUE}🏷️  Tagging image as: $IMAGE_URI${NC}"
docker tag $IMAGE_NAME $IMAGE_URI

# Push to ECR
echo -e "${BLUE}🚀 Pushing image to ECR...${NC}"
docker push $IMAGE_URI

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image pushed to ECR successfully!${NC}"
else
    echo -e "${RED}❌ Image push failed${NC}"
    exit 1
fi

# Test the image locally (optional)
echo -e "${BLUE}🧪 Testing image locally...${NC}"
echo "Testing Tesseract installation..."
docker run --rm $IMAGE_NAME tesseract --version

echo ""
echo -e "${GREEN}🎉 SUCCESS! Custom SageMaker container is ready!${NC}"
echo "=================================================="
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Update your Terraform configuration to use the new image:"
echo "   Image URI: $IMAGE_URI"
echo ""
echo "2. Add this to your variables.tf:"
echo "   variable \"custom_inference_image\" {"
echo "     default = \"$IMAGE_URI\""
echo "   }"
echo ""
echo "3. Update sagemaker_endpoint.tf to use the custom image"
echo ""
echo -e "${BLUE}💡 The image includes:${NC}"
echo "   ✅ Tesseract OCR (multiple languages)"
echo "   ✅ pytesseract Python wrapper"  
echo "   ✅ PIL/Pillow for image processing"
echo "   ✅ OpenCV for advanced image operations"
echo "   ✅ PDF2Image for PDF processing"
echo "   ✅ All original PyTorch capabilities"
