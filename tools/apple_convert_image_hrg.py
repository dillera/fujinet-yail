#!/usr/bin/env python

import sys
import numpy as np
from PIL import Image

def convert_to_hrg(input_image, output_file):
    # Load the image
    img = Image.open(input_image).convert('L')  # Convert to grayscale
    img = img.resize((280, 192))  # Resize to Apple IIc HiRes resolution

    # Convert image to binary format (1 bit per pixel)
    data = np.array(img)
    threshold = 128  # Adjust as needed
    binary_data = (data > threshold).astype(np.uint8)

    # Pack the binary data into bytes
    packed_data = np.packbits(binary_data, axis=-1)

    # Write the raw data to file
    with open(output_file, 'wb') as f:
        f.write(packed_data)

if __name__ == "__main__":
   
    input_image = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_hrg(input_image, output_file)

#done
