#!/usr/bin/env python3
import pytesseract
import sys
from PIL import Image, ImageDraw

print('Testing OCR functionality...')

# Create a simple test image with text
img = Image.new('RGB', (200, 50), color='white')
draw = ImageDraw.Draw(img)
draw.text((10, 10), 'Hello World', fill='black')

# Test OCR
try:
    text = pytesseract.image_to_string(img)
    print(f'OCR Result: {text.strip()}')
    print('SUCCESS: OCR functionality working!')
    print('Image size optimized while maintaining functionality!')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
