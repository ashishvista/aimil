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
mkdir -p "$LAYER_DIR/python/lib/$PYTHON_VERSION/site-packages"

# Check if requirements.txt exists
if [ ! -f "lambda/requirements.txt" ]; then
    echo "❌ requirements.txt not found in lambda/ directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r lambda/requirements.txt -t "$LAYER_DIR/python/lib/$PYTHON_VERSION/site-packages" --no-deps --quiet

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
ls -la "$LAYER_DIR/python/lib/$PYTHON_VERSION/site-packages/" | head -10
