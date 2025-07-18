#!/bin/bash

# Test script for OCR Pipeline
# This script tests the deployed OCR pipeline endpoints

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get API Gateway URL from Terraform output
get_api_url() {
    if [ -f "deployment_info.txt" ]; then
        API_URL=$(grep "API Gateway URL:" deployment_info.txt | cut -d' ' -f4)
    else
        API_URL=$(terraform output -raw api_gateway_url 2>/dev/null || echo "")
    fi
    
    if [ -z "$API_URL" ]; then
        print_error "Could not find API Gateway URL. Make sure the infrastructure is deployed."
        exit 1
    fi
    
    echo "$API_URL"
}

# Test presigned URL generation
test_presigned_url() {
    print_test "Testing presigned URL generation..."
    
    local response=$(curl -s -X POST "$API_URL/prod/upload" \
        -H "Content-Type: application/json" \
        -d '{
            "file_extension": "jpg",
            "content_type": "image/jpeg"
        }')
    
    echo "Response:"
    echo "$response" | jq . 2>/dev/null || echo "$response"
    
    # Extract upload URL for next test
    UPLOAD_URL=$(echo "$response" | jq -r '.upload_url' 2>/dev/null || echo "")
    FILE_KEY=$(echo "$response" | jq -r '.file_key' 2>/dev/null || echo "")
    
    if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
        print_status "✓ Presigned URL generation test passed"
        return 0
    else
        print_error "✗ Presigned URL generation test failed"
        return 1
    fi
}

# Create a test image
create_test_image() {
    print_test "Creating test image..."
    
    # Create a simple test image with text using ImageMagick (if available)
    if command -v convert &> /dev/null; then
        convert -size 400x100 xc:white -font Arial -pointsize 20 -gravity center \
            -draw "text 0,0 'Hello OCR World!'" test_image.jpg
        print_status "✓ Test image created with ImageMagick"
    elif command -v python3 &> /dev/null; then
        # Create test image using Python PIL
        python3 << 'EOF'
from PIL import Image, ImageDraw, ImageFont
import sys

try:
    # Create a white image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, 30), "Hello OCR World!", fill='black', font=font)
    
    # Save image
    img.save('test_image.jpg')
    print("Test image created successfully")
except Exception as e:
    print(f"Error creating test image: {e}")
    sys.exit(1)
EOF
        print_status "✓ Test image created with Python PIL"
    else
        print_warning "Neither ImageMagick nor Python PIL available. Please provide test_image.jpg manually."
        return 1
    fi
    
    return 0
}

# Test file upload
test_file_upload() {
    if [ -z "$UPLOAD_URL" ] || [ -z "$FILE_KEY" ]; then
        print_error "Missing upload URL or file key from previous test"
        return 1
    fi
    
    if [ ! -f "test_image.jpg" ]; then
        print_error "Test image not found. Run create_test_image first."
        return 1
    fi
    
    print_test "Testing file upload..."
    
    local upload_response=$(curl -s -X PUT "$UPLOAD_URL" \
        -H "Content-Type: image/jpeg" \
        --data-binary @test_image.jpg \
        -w "%{http_code}")
    
    if [[ "$upload_response" == *"200" ]]; then
        print_status "✓ File upload test passed"
        return 0
    else
        print_error "✗ File upload test failed (HTTP: $upload_response)"
        return 1
    fi
}

# Test OCR processing
test_ocr_processing() {
    if [ -z "$FILE_KEY" ]; then
        print_error "Missing file key from previous test"
        return 1
    fi
    
    print_test "Testing OCR processing..."
    
    local response=$(curl -s -X POST "$API_URL/prod/process" \
        -H "Content-Type: application/json" \
        -d "{
            \"s3_key\": \"$FILE_KEY\",
            \"method\": \"tesseract\"
        }")
    
    echo "OCR Response:"
    echo "$response" | jq . 2>/dev/null || echo "$response"
    
    # Check if we got a valid response
    if echo "$response" | jq -e '.tesseract' &>/dev/null; then
        print_status "✓ OCR processing test passed"
        return 0
    else
        print_error "✗ OCR processing test failed"
        return 1
    fi
}

# Test with base64 encoded image
test_base64_processing() {
    if [ ! -f "test_image.jpg" ]; then
        print_warning "Test image not found. Skipping base64 test."
        return 0
    fi
    
    print_test "Testing base64 image processing..."
    
    # Encode image to base64
    local base64_image=$(base64 -i test_image.jpg)
    
    local response=$(curl -s -X POST "$API_URL/prod/process" \
        -H "Content-Type: application/json" \
        -d "{
            \"image_base64\": \"$base64_image\",
            \"method\": \"both\"
        }")
    
    echo "Base64 OCR Response:"
    echo "$response" | jq . 2>/dev/null || echo "$response"
    
    # Check if we got a valid response
    if echo "$response" | jq -e '.tesseract' &>/dev/null; then
        print_status "✓ Base64 processing test passed"
        return 0
    else
        print_error "✗ Base64 processing test failed"
        return 1
    fi
}

# Cleanup test files
cleanup() {
    print_status "Cleaning up test files..."
    rm -f test_image.jpg
    print_status "✓ Cleanup complete"
}

# Main test function
main() {
    echo "OCR Pipeline Test Suite"
    echo "======================"
    
    # Get API URL
    API_URL=$(get_api_url)
    print_status "Using API URL: $API_URL"
    
    # Check dependencies
    if ! command -v jq &> /dev/null; then
        print_warning "jq is not installed. JSON responses will not be formatted."
    fi
    
    local failed_tests=0
    
    # Run tests
    echo ""
    print_status "Starting test suite..."
    
    if ! test_presigned_url; then
        ((failed_tests++))
    fi
    
    echo ""
    if create_test_image; then
        if ! test_file_upload; then
            ((failed_tests++))
        fi
        
        echo ""
        if ! test_ocr_processing; then
            ((failed_tests++))
        fi
        
        echo ""
        if ! test_base64_processing; then
            ((failed_tests++))
        fi
    else
        print_warning "Skipping upload and processing tests due to missing test image"
    fi
    
    # Cleanup
    echo ""
    cleanup
    
    # Summary
    echo ""
    echo "Test Summary"
    echo "============"
    if [ $failed_tests -eq 0 ]; then
        print_status "🎉 All tests passed!"
    else
        print_error "❌ $failed_tests test(s) failed"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--create-image|--cleanup|--help]"
        echo ""
        echo "Options:"
        echo "  --create-image    Only create test image"
        echo "  --cleanup         Only cleanup test files"
        echo "  --help            Show this help"
        exit 0
        ;;
    --create-image)
        create_test_image
        exit $?
        ;;
    --cleanup)
        cleanup
        exit 0
        ;;
    *)
        main
        ;;
esac
