#!/bin/bash

# Build Lambda Layer for Python dependencies using Docker only
# This script creates a Lambda layer with all required Python packages

set -e

# Configuration
LAYER_NAME="ocr-pipeline-dependencies"
PYTHON_VERSION="python3.9"
LAYER_DIR="lambda_layer_temp"
ZIP_FILE="lambda_layer.zip"

echo "🏗️  Building Lambda Layer: $LAYER_NAME"

# Clean up previous builds
if [ -d "$LAYER_DIR" ]; then
    echo "🧹 Cleaning up previous build..."
    rm -rf "$LAYER_DIR"
fi

if [ -f "$ZIP_FILE" ]; then
    rm -f "$ZIP_FILE"
fi

# Create layer directory structure
echo "📁 Creating layer directory structure..."
mkdir -p "$LAYER_DIR/python/lib/python3.9/site-packages"

# Check if requirements.txt exists
if [ ! -f "lambda/requirements.txt" ]; then
    echo "❌ requirements.txt not found in lambda/ directory"
    exit 1
fi

# Install dependencies using Docker for Linux compatibility
echo "🐳 Using Docker to build Linux-compatible packages..."
echo "📦 Installing Python dependencies with full system libraries..."

# Check if Docker is available
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found! This script requires Docker to build Linux-compatible packages."
    echo "Please install Docker and try again."
    exit 1
fi

# Use Amazon Linux 2 image which matches Lambda runtime (x86_64)
docker run --rm --platform linux/amd64 \
    -v "$(pwd)/lambda/requirements.txt:/tmp/requirements.txt" \
    -v "$(pwd)/$LAYER_DIR:/tmp/layer" \
    amazonlinux:2 \
    bash -c "
        echo '🔄 Updating system packages...'
        yum update -y && 
        
        echo '📦 Installing system dependencies for PIL/Pillow...'
        yum install -y python3 python3-pip python3-devel gcc gcc-c++ make \
                      libjpeg-devel zlib-devel libtiff-devel freetype-devel \
                      lcms2-devel libwebp-devel tcl-devel tk-devel && 
        
        echo '🔧 Upgrading pip and build tools...'
        pip3 install --upgrade pip setuptools wheel && 
        
        echo '📦 Installing Python packages with native compilation...'
        pip3 install -r /tmp/requirements.txt -t /tmp/layer/python/lib/python3.9/site-packages --no-cache-dir &&
        
        echo '✅ Package installation complete!'
        echo '📋 Installed packages:'
        ls -la /tmp/layer/python/lib/python3.9/site-packages/ | head -10
    "

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

# Remove unnecessary files to reduce layer size
echo "🧹 Removing unnecessary files..."
find "$LAYER_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "*.pyo" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "test" -type d -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Verify PIL native libraries are present
echo "🔍 Verifying PIL native libraries..."
if [ -f "$LAYER_DIR/python/lib/python3.9/site-packages/PIL/_imaging.cpython-39m-x86_64-linux-gnu.so" ]; then
    echo "✅ PIL native libraries found:"
    ls -la "$LAYER_DIR/python/lib/python3.9/site-packages/PIL/"*imaging*.so
elif [ -f "$LAYER_DIR/python/lib/python3.9/site-packages/PIL/_imaging.cpython-37m-x86_64-linux-gnu.so" ]; then
    echo "✅ PIL native libraries found (Python 3.7):"
    ls -la "$LAYER_DIR/python/lib/python3.9/site-packages/PIL/"*imaging*.so
else
    echo "⚠️  PIL native libraries not found, checking for alternative naming..."
    ls -la "$LAYER_DIR/python/lib/python3.9/site-packages/PIL/" | grep -i imaging || echo "❌ No PIL imaging libraries found!"
fi

echo "✅ Lambda layer built successfully: $LAYER_DIR"
echo "📊 Layer size: $(du -sh $LAYER_DIR | cut -f1)"

# List some contents for verification
echo "📋 Layer contents summary:"
echo "Total files: $(find $LAYER_DIR -type f | wc -l)"
echo "Main packages:"
ls -la "$LAYER_DIR/python/lib/python3.9/site-packages/" | grep -E "^d" | head -5

# Create ZIP file for layer
echo "📦 Creating ZIP file..."
cd "$LAYER_DIR"
zip -r "../$ZIP_FILE" . -q
cd ..

echo "✅ Lambda layer ZIP created: $ZIP_FILE"
echo "📊 ZIP file size: $(du -sh $ZIP_FILE | cut -f1)"

# Clean up temp directory
rm -rf "$LAYER_DIR"

echo "🎉 Lambda layer build complete!"
echo "📁 Layer ZIP: $ZIP_FILE"
echo ""
echo "Next steps:"
echo "1. Deploy the infrastructure: ./terraform_deploy.sh"
echo "2. Test the API endpoints"
