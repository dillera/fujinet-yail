#!/usr/bin/env python


import sys
from PIL import Image

# Define the Apple II HRG palette
palette = [
    (0, 0, 0),       # Black
    (114, 38, 64),   # Deep Red
    (64, 51, 127),   # Dark Blue
    (255, 0, 255),   # Purple
    (38, 89, 64),    # Dark Green
    (128, 128, 128), # Grey
    (0, 255, 255),   # Light Blue
    (191, 255, 191), # Light Green
    (255, 64, 0),    # Orange
    (255, 255, 0),   # Yellow
    (255, 255, 255)  # White
]

def closest_color(rgb):
    r, g, b = rgb
    color_diffs = []
    for color in palette:
        cr, cg, cb = color
        color_diff = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]

# Convert image to Apple II palette
input_path = sys.argv[1]
output_path = sys.argv[2]
image = Image.open(input_path).convert('RGB')
converted_image = Image.new('RGB', image.size)

for y in range(image.height):
    for x in range(image.width):
        converted_image.putpixel((x, y), closest_color(image.getpixel((x, y))))

# Save as Apple II HRG format
data = bytearray()

# Add the header (AppleSingle file format header example)
data.extend([0x00, 0x05, 0x16, 0x00])

# Add the image data
for y in range(0, 192, 8):
    for x in range(280):
        byte = 0
        for bit in range(8):
            color = converted_image.getpixel((x, y + bit))
            byte |= (palette.index(color) & 0x01) << bit
        data.append(byte)

# Write to file
with open(output_path, 'wb') as f:
    f.write(data)
