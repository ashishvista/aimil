#!/bin/bash

# Build Lambda Layer for Python dependencies
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
mkdir -p "$LAYER_DIR/python"

# Check if requirements.txt exists
if [ ! -f "lambda/requirements.txt" ]; then
    echo "❌ requirements.txt not found in lambda/ directory"
    exit 1
fi

# Install dependencies using Docker for Linux compatibility
echo "📦 Installing Python dependencies using Docker..."
if command -v docker >/dev/null 2>&1; then
    echo "🐳 Using Docker to build Linux-compatible packages..."
    # Use Amazon Linux 2 image which matches Lambda runtime (x86_64)
    docker run --rm --platform linux/amd64 \
        -v "$(pwd)/lambda/requirements.txt:/tmp/requirements.txt" \
        -v "$(pwd)/$LAYER_DIR:/tmp/layer" \
        amazonlinux:2 \
        bash -c "
            yum update -y && 
            yum install -y python3 python3-pip && 
            pip3 install -r /tmp/requirements.txt -t /tmp/layer/python --no-deps
        "
else
    echo "⚠️  Docker not found, falling back to platform-independent packages"
    echo "📦 Installing Python dependencies with platform-independent flag..."
    pip3 install -r lambda/requirements.txt -t "$LAYER_DIR/python" --no-deps --only-binary=:all: --platform linux_x86_64 --quiet 2>/dev/null || \
    pip3 install -r lambda/requirements.txt -t "$LAYER_DIR/python" --no-deps --quiet
fi

# Remove unnecessary files to reduce layer size
echo "🧹 Removing unnecessary files..."
find "$LAYER_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "*.pyo" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Lambda layer built successfully: $LAYER_DIR"
echo "📊 Layer size: $(du -sh $LAYER_DIR | cut -f1)"

# List some contents for verification
echo "📋 Sample layer contents:"
ls -la "$LAYER_DIR/python/" | head -10

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
