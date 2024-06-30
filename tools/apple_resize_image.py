#!/usr/bin/env python

import sys
from PIL import Image

# Load the image
input_path = sys.argv[1]
output_path = sys.argv[2]
image = Image.open(input_path)

# Resize the image
resized_image = image.resize((280, 192), Image.LANCZOS)
resized_image.save(output_path, format='JPEG')
