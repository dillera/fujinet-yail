#!/usr/bin/env python

import os
import argparse
from typing import List, Union, Callable
import requests
import re
import time
import logging
import os
from tqdm import tqdm
import socket
from threading import Thread, Lock
import random
from duckduckgo_search import DDGS
from fastcore.all import *
from pprint import pformat
from PIL import Image
import numpy as np
import sys
import openai

# Set up logging first thing
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

SOCKET_WAIT_TIME = 1
GRAPHICS_8 = 2
GRAPHICS_9 = 4
GRAPHICS_VBXE_1 = 0x11
GRAPHICS_RANDOM = 42
YAIL_W = 320
YAIL_H = 220

DL_BLOCK = 0x04
XDL_BLOCK = 0x05
PALETTE_BLOCK = 0x06
IMAGE_BLOCK = 0x07


# The yail_data will contain the image that is to be sent.  It
# is protected with a Mutex so that when the image is being sent
# it won't be written by the server.
mutex = Lock()
yail_data = None
connections = 0
camera_thread = None
camera_done = False
filenames = []
camera_name = None
openai_model = None
openai_size = None
openai_quality = None
openai_style = None

def prep_image_for_vbxe(image: Image.Image, target_width: int=YAIL_W, target_height: int=YAIL_H) -> Image.Image:
    logger.info(f'Image size: {image.size}')

    # Calculate the new size preserving the aspect ratio
    image_ratio = image.width / image.height
    target_ratio = target_width / target_height

    if image_ratio > target_ratio:
        # Image is wider than target, fit to width
        new_width = target_width
        new_height = int(target_width / image_ratio)
    else:
        # Image is taller than target, fit to height
        new_width = int(target_height * image_ratio)
        new_height = target_height

    # Resize the image
    image = image.resize((new_width, new_height), Image.BILINEAR)
    logger.info(f'Image new size: {image.size}')

    # Create a new image with the target size and a black background
    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    # Calculate the position to paste the resized image onto the black background
    paste_x = (target_width - image.width) // 2
    paste_y = (target_height - image.height) // 2

    # Paste the resized image onto the black background
    new_image.paste(image, (paste_x, paste_y))

    # Replace the original image with the new image
    return new_image


def fix_aspect(image: Image.Image, crop: bool=False) -> Image.Image:
    aspect = YAIL_W/YAIL_H   # YAIL aspect ratio
    aspect_i = 1/aspect
    w = image.size[0]
    h = image.size[1]
    img_aspect = w/h

    if crop:
        if img_aspect > aspect:  # wider than YAIL aspect
            new_width = int(h * aspect)
            new_width_diff = w - new_width
            new_width_diff_half = int(new_width_diff/2)
            image = image.crop((new_width_diff_half, 0, w-new_width_diff_half, h))
        else:                    # taller than YAIL aspect
            new_height = int(w * aspect_i)
            new_height_diff = h - new_height
            new_height_diff_half = int(new_height_diff/2)
            image = image.crop((0, new_height_diff_half, w, h-new_height_diff_half))
    else:
        if img_aspect > aspect:  # wider than YAIL aspect
            new_height = int(w * aspect_i)
            background = Image.new("L", (w,new_height))
            background.paste(image, (0, int((new_height-h)/2)))
            image = background
        else:                    # taller than YAIL aspect
            new_width = int(h * aspect)
            background = Image.new("L", (new_width, h))
            background.paste(image, (int((new_width-w)/2), 0))
            image = background

    return image

def dither_image(image: Image.Image) -> Image.Image:
    return image.convert('1')

def pack_bits(image: Image.Image) -> np.ndarray:
    bits = np.array(image)
    return np.packbits(bits, axis=1)

def pack_shades(image: Image.Image) -> np.ndarray:
    yail = image.resize((int(YAIL_W/4),YAIL_H), Image.LANCZOS)
    yail = yail.convert(dither=Image.FLOYDSTEINBERG, colors=16)

    im_matrix = np.array(yail)
    im_values = im_matrix[:,:]

    evens = im_values[:,::2]
    odds = im_values[:,1::2]

    # Each byte holds 2 pixels.  The upper four bits for the left pixel and the lower four bits for the right pixel.
    evens_scaled = (evens >> 4) << 4 # left pixel
    odds_scaled =  (odds >> 4)       # right pixel

    # Combine the two 4bit values into a single byte
    combined = evens_scaled + odds_scaled
    
    return combined.astype('int8')

def show_dithered(image: Image.Image) -> None:
    image.show()

def show_shades(image_data: np.ndarray) -> None:
    pil_image_yai = Image.fromarray(image_data, mode='L')
    pil_image_yai.resize((320,220), resample=None).show()

def convertToYai(image_data: bytearray, gfx_mode: int) -> bytearray:
    import struct

    ttlbytes = image_data.shape[0] * image_data.shape[1]

    image_yai = bytearray()
    image_yai += bytes([1, 1, 0])            # version
    image_yai += bytes([gfx_mode])           # Gfx mode (8,9)
    image_yai += bytes([3])                  # Memory block type
    image_yai += struct.pack("<H", ttlbytes) # num bytes height x width
    image_yai += bytearray(image_data)       # image

    return image_yai

def convertToYaiVBXE(image_data: bytes, palette_data: bytes, gfx_mode: int) -> bytearray:
    import struct

    #ttlbytes = YAIL_W * YAIL_H; # image_data.shape[0] * image_data.shape[1]
    logger.info('Image data size: %d' % len(image_data))
    logger.info('Palette data size: %d' % len(palette_data))

    image_yai = bytearray()
    image_yai += bytes([1, 4, 0])            # version
    image_yai += bytes([gfx_mode])           # Gfx mode (8,9)
    image_yai += struct.pack("<B", 2)        # number of memory blocks
    image_yai += bytes([PALETTE_BLOCK])             # Memory block type
    image_yai += struct.pack("<I", len(palette_data)) # palette size
    image_yai += bytearray(palette_data)  # palette
    image_yai += bytes([IMAGE_BLOCK])                  # Memory block type
    image_yai += struct.pack("<I", len(image_data)) # num bytes height x width
    image_yai += bytearray(image_data)       # image

    logger.info('YAI size: %d' % len(image_yai))

    return image_yai

def update_yail_data(data: np.ndarray, gfx_mode: int, thread_safe: bool = True) -> None:
    global yail_data
    if thread_safe:
        mutex.acquire()
    try:
        yail_data = convertToYai(data, gfx_mode)
    finally:
        if thread_safe:
            mutex.release()

def send_yail_data(client_socket: socket.socket, thread_safe: bool=True) -> None:
    global yail_data

    if thread_safe:
        mutex.acquire()
    try:
        data = yail_data   # a local copy
    finally:
        if thread_safe:
            mutex.release()

    if data is not None:
        client_socket.sendall(data)
        logger.info('Sent YAIL data')

def stream_YAI(client: str, gfx_mode: int, url: str = None, filepath: str = None) -> bool:
    from io import BytesIO

    global YAIL_H

    # download the body of response by chunk, not immediately
    try:
        if url is not None:
            logger.info(f'Loading %s %s' % (url, url.encode()))

            file_size = 0

            response = requests.get(url, stream=True, timeout=30)

            # get the file name
            filepath = ''
            exts = ['.jpg', '.jpeg', '.gif', '.png']
            ext = re.findall('|'.join(exts), url)
            if len(ext):
                pos_ext = url.find(ext[0])
                if pos_ext >= 0:
                    pos_name = url.rfind("/", 0, pos_ext)
                    filepath =  url[pos_name+1:pos_ext+4]

            # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
            image_data = b''
            progress = tqdm(response.iter_content(1024), f"Downloading {filepath}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
            for data in progress:
                # collect all the data
                image_data += data

                # update the progress bar manually
                progress.update(len(data))

            image_bytes_io = BytesIO()
            image_bytes_io.write(image_data)
            image = Image.open(image_bytes_io)

        elif filepath is not None:
            image = Image.open(filepath)

        if gfx_mode == GRAPHICS_8 or gfx_mode == GRAPHICS_9:
            gray = image.convert(mode='L')
            gray = fix_aspect(gray)
            gray = gray.resize((YAIL_W,YAIL_H), Image.LANCZOS)

            if gfx_mode == GRAPHICS_8:
                gray_dithered = dither_image(gray)
                image_data = pack_bits(gray_dithered)
            elif gfx_mode == GRAPHICS_9:
                image_data = pack_shades(gray)

            image_yai = convertToYai(image_data, gfx_mode)

        else:  # VBXE mode
            # Make the image fit out screen format but preserve it's aspect ratio
            image_resized = prep_image_for_vbxe(image, target_width=320, target_height=240)
            # Convert the image to use a palette
            image_resized = image_resized.convert('P', palette=Image.ADAPTIVE, colors=256)
            logger.info(f'Image size: {image_resized.size}')
            #image_resized.show()
            # Get the palette
            palette = image_resized.getpalette()
            # Get the image data
            image_resized = image_resized.tobytes()
            logger.info(f'Image data size: {len(image_resized)}')
            # Offset the palette entries by one
            offset_palette = [0] * 3 + palette[:-3]
            # Offset the image data by one
            offset_image_data = bytes((byte + 1) % 256 for byte in image_resized)

            image_yai = convertToYaiVBXE(offset_image_data, offset_palette, gfx_mode)

        client.sendall(image_yai)

        return True

    except Exception as e:
        logger.error(f'Exception: {e} **{file_size}')
        return False

# This uses the DuckDuckGo search engine to find images.  This is handled by the duckduckgo_search package.
def search_images(term: str, max_images: int=1000) -> list:
    logger.info(f"Searching for '{term}'")
    with DDGS() as ddgs:
        results = L([r for r in ddgs.images(term, max_results=max_images)])

        urls = []
        for result in results:
            urls.append(result['image'])

        return urls

def generate_image_with_openai(prompt: str, api_key: str = None, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard", style: str = "vivid") -> str:
    """
    Generate an image using OpenAI's image generation models and return the URL.
    
    Args:
        prompt (str): The text prompt to generate an image from
        api_key (str, optional): OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        model (str, optional): The model to use. Options: "dall-e-3" (default) or "gpt-4o"
        size (str, optional): Image size. Options for DALL-E 3: "1024x1024" (default), "1792x1024", or "1024x1792"
                             For GPT-4o, size parameter may not be applicable
        quality (str, optional): Image quality. Options: "standard" (default) or "hd" (DALL-E 3 only)
        style (str, optional): Image style. Options: "vivid" (default) or "natural" (DALL-E 3 only)
        
    Returns:
        str: URL of the generated image
    """
    try:
        # Set API key from parameter or environment variable
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
            return None
        
        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Generate image based on model type
        logger.info(f"Generating image with {model} model, prompt: '{prompt}'")
        
        if model.lower() == "gpt-4o":
            # GPT-4o image generation
            # Note: Parameters may differ for GPT-4o, adjust as needed based on official documentation
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an image generation assistant. Generate an image based on the user's description."},
                    {"role": "user", "content": f"Generate an image of: {prompt}"}
                ],
                tools=[{"type": "image_generator"}]
            )
            
            # Extract image URL from response
            # Note: This is a placeholder implementation and may need adjustment based on actual API response format
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "image_generator":
                    # Parse the function arguments to get the image URL
                    import json
                    function_args = json.loads(tool_call.function.arguments)
                    image_url = function_args.get("url")
                    if image_url:
                        logger.info(f"Image generated successfully with GPT-4o: {image_url}")
                        return image_url
            
            logger.error("Failed to extract image URL from GPT-4o response")
            return None
            
        else:  # Default to DALL-E 3
            # DALL-E 3 image generation
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=size,
                quality=quality,
                style=style
            )
            
            # Return the URL of the generated image
            image_url = response.data[0].url
            logger.info(f"Image generated successfully with DALL-E 3: {image_url}")
            return image_url
        
    except Exception as e:
        logger.error(f"Error generating image with OpenAI: {e}")
        return None

def camera_handler(gfx_mode: int) -> None:
    import pygame.camera
    import pygame.image

    SHOW_WEBCAM_VIEW = False

    global camera_done

    pygame.camera.init()

    if camera_name is not None:
        webcam = pygame.camera.Camera(camera_name)

        webcam.start()
    else:
        cameras = pygame.camera.list_cameras()

        # going to try each camera in the list until we have one
        for camera in cameras:
            try:
                logger.info("Trying camera %s ..." % camera)

                webcam = pygame.camera.Camera(camera) #'/dev/video60') #cameras[0])

                webcam.start()
            except Exception as ex:
                logger.warn("Unable to use camera %s ..." % camera)

    # grab first frame
    img = webcam.get_image()

    WIDTH = img.get_width()
    HEIGHT = img.get_height()

    if SHOW_WEBCAM_VIEW:
        screen = pygame.display.set_mode( ( WIDTH, HEIGHT ) )
        pygame.display.set_caption("pyGame Camera View")

    while not camera_done:
        if SHOW_WEBCAM_VIEW:
            for e in pygame.event.get() :
                if e.type == pygame.QUIT :
                    sys.exit()

        imgdata = pygame.surfarray.array3d(img)
        imgdata = imgdata.swapaxes(0,1)
        pil_image = Image.fromarray(np.array(imgdata))
        gray = pil_image.convert(mode='L')
        gray = fix_aspect(gray, crop=True)
        gray = gray.resize((YAIL_W,YAIL_H), Image.LANCZOS)

        if gfx_mode == GRAPHICS_8:
            gray = dither_image(gray)
            update_yail_data(pack_bits(gray))
        elif gfx_mode == GRAPHICS_9:
            update_yail_data(pack_shades(gray))

        # draw frame
        if SHOW_WEBCAM_VIEW:
            screen.blit(img, (0,0))
            pygame.display.flip()

        # grab next frame    
        img = webcam.get_image()

    camera_done = False

def handle_client_connection(client_socket: socket.socket) -> None:
    global connections
    global camera_thread
    global camera_done
    global YAIL_H
    global openai_model
    global openai_size
    global openai_quality
    global openai_style
    
    gfx_mode = GRAPHICS_8
    client_mode = None
    last_prompt = None  # Store the last prompt for regeneration

    connections = connections + 1
    logger.info(f'Starting Connection: %d' % connections)

    try:
        done = False
        url_idx = 0
        tokens = []
        while not done:
            if len(tokens) == 0:
                request = client_socket.recv(1024)
                logger.info('Client request ' + pformat(request))
                r_string = request.decode('UTF-8')
                tokens = r_string.rstrip(' \r\n').split(' ')
            logger.info('Tokens ' + pformat(tokens))

            if tokens[0] == 'video':
                client_mode = 'video'
                if camera_thread is None:
                    camera_done = False
                    camera_thread = Thread(target=camera_handler)
                    camera_thread.daemon = True
                    camera_thread.start()
                send_yail_data(client_socket)
                tokens.pop(0)

            elif tokens[0] == 'search':
                client_mode = 'search'
                urls = search_images(' '.join(tokens[1:]))
                url_idx = random.randint(0, len(urls)-1)
                url = urls[url_idx]
                if url:
                    # Loop if we have a problem with the image, selecting the next.
                    while not stream_YAI(client_socket, gfx_mode, url=url):
                        logger.warning(f'Problem with %s trying another...', url)
                        url_idx = random.randint(0, len(urls)-1)
                        url = urls[url_idx]
                        time.sleep(SOCKET_WAIT_TIME)
                tokens = []
                
            elif tokens[0] == 'generate':
                client_mode = 'generate'
                # Join all tokens after 'generate' as the prompt
                prompt = ' '.join(tokens[1:])
                last_prompt = prompt  # Store the prompt for later use with 'next' command
                logger.info(f"Generating image with prompt: '{prompt}'")
                
                # Generate image using OpenAI
                url = generate_image_with_openai(prompt, model=openai_model, size=openai_size, quality=openai_quality, style=openai_style)
                
                if url:
                    # Stream the generated image to the client
                    if not stream_YAI(client_socket, gfx_mode, url=url):
                        logger.warning(f'Problem with generated image: {url}')
                        client_socket.send(bytes(b'ERROR: Failed to stream generated image'))
                else:
                    logger.warning('Failed to generate image with OpenAI')
                    client_socket.send(bytes(b'ERROR: Failed to generate image'))
                
                tokens = []

            elif tokens[0] == 'files':
                client_mode = 'files'
                if len(filenames) > 0:
                    file_idx = random.randint(0, len(filenames)-1)
                    filename = filenames[file_idx]
                    if filename:
                        # Loop if we have a problem with the image, selecting the next.
                        while not stream_YAI(client_socket, gfx_mode, filepath=filename):
                            logger.warning(f'Problem with %s trying another...', filename)
                            file_idx = random.randint(0, len(filenames)-1)
                            filename = filenames[file_idx]
                            time.sleep(SOCKET_WAIT_TIME)
                tokens.pop(0)

            elif tokens[0] == 'next':
                if client_mode == 'search':
                    url = None
                    url_idx = random.randint(0, len(urls)-1)
                    url = urls[url_idx]
                    if url:
                        # Loop if we have a problem with the image, selecting the next.
                        while not stream_YAI(client_socket, gfx_mode, url=url):
                            logger.warning('Problem with image trying another...')
                            url_idx = random.randint(0, len(urls)-1)
                            url = urls[url_idx]
                            time.sleep(SOCKET_WAIT_TIME)
                    tokens.pop(0)
                elif client_mode == 'video':
                    send_yail_data(client_socket)
                    tokens.pop(0)
                elif client_mode == 'generate':
                    # For generate mode, we'll regenerate with the same prompt
                    # The prompt is stored in last_prompt
                    prompt = last_prompt
                    logger.info(f"Regenerating image with prompt: '{prompt}'")
                    
                    # Generate a new image with the same prompt
                    url = generate_image_with_openai(prompt, model=openai_model, size=openai_size, quality=openai_quality, style=openai_style)
                    
                    if url:
                        # Stream the generated image to the client
                        if not stream_YAI(client_socket, gfx_mode, url=url):
                            logger.warning(f'Problem with generated image: {url}')
                            client_socket.send(bytes(b'ERROR: Failed to stream generated image'))
                    else:
                        logger.warning('Failed to generate image with OpenAI')
                        client_socket.send(bytes(b'ERROR: Failed to generate image'))
                    
                    tokens.pop(0)
                elif client_mode == 'files':
                    filename = None
                    if len(filenames) > 0:
                        file_idx = random.randint(0, len(filenames)-1)
                        filename = filenames[file_idx]
                        if filename:
                            # Loop if we have a problem with the image, selecting the next.
                            while not stream_YAI(client_socket, gfx_mode, filepath=filename):
                                logger.warning(f'Problem with %s trying another...', filename)
                                file_idx = random.randint(0, len(filenames)-1)
                                filename = filenames[file_idx]
                                time.sleep(SOCKET_WAIT_TIME)
                    tokens.pop(0)

            elif tokens[0] == 'gfx':
                tokens.pop(0)
                gfx_mode = int(tokens[0])
                #if gfx_mode > GRAPHICS_9:  # VBXE
                #    global YAIL_H
                #    YAIL_H = 240
                tokens.pop(0)

            elif tokens[0] == 'openai-config':
                tokens.pop(0)
                if len(tokens) > 0:
                    # Process OpenAI configuration parameters
                    
                    # Format: openai-config [param] [value]
                    param = tokens[0].lower()
                    tokens.pop(0)
                    
                    if len(tokens) > 0:
                        value = tokens[0]
                        tokens.pop(0)
                        
                        if param == "model":
                            if value in ["dall-e-3", "gpt-4o"]:
                                openai_model = value
                                client_socket.send(bytes(f"OpenAI model set to {value}".encode('utf-8')))
                            else:
                                client_socket.send(bytes(f"Invalid model. Use 'dall-e-3' or 'gpt-4o'".encode('utf-8')))
                        
                        elif param == "size":
                            if value in ["1024x1024", "1792x1024", "1024x1792"]:
                                openai_size = value
                                client_socket.send(bytes(f"Image size set to {value}".encode('utf-8')))
                            else:
                                client_socket.send(bytes(f"Invalid size. Use '1024x1024', '1792x1024', or '1024x1792'".encode('utf-8')))
                        
                        elif param == "quality":
                            if value in ["standard", "hd"]:
                                openai_quality = value
                                client_socket.send(bytes(f"Image quality set to {value}".encode('utf-8')))
                            else:
                                client_socket.send(bytes(f"Invalid quality. Use 'standard' or 'hd'".encode('utf-8')))
                        
                        elif param == "style":
                            if value in ["vivid", "natural"]:
                                openai_style = value
                                client_socket.send(bytes(f"Image style set to {value}".encode('utf-8')))
                            else:
                                client_socket.send(bytes(f"Invalid style. Use 'vivid' or 'natural'".encode('utf-8')))
                        
                        else:
                            client_socket.send(bytes(f"Unknown parameter '{param}'. Use 'model', 'size', 'quality', or 'style'".encode('utf-8')))
                    else:
                        client_socket.send(bytes(f"Current OpenAI config: model={openai_model}, size={openai_size}, quality={openai_quality}, style={openai_style}".encode('utf-8')))
                else:
                    client_socket.send(bytes(f"Current OpenAI config: model={openai_model}, size={openai_size}, quality={openai_quality}, style={openai_style}".encode('utf-8')))

            elif tokens[0] == 'quit':
                done = True
                tokens.pop(0)

            else:
                logger.info('Received {}'.format(r_string.rstrip(' \r\n')))
                client_socket.send(bytes(b'ACK!'))

    except Exception as ex:
        logger.critical('Problem handling client ' + str(ex))

    finally:
        client_socket.close()
        logger.info(f'Closing Connection: %d' % connections)
        connections = connections - 1
        if connections == 0:   # Maybe should look into killing this thread when there are no video connections.
            camera_done = True
            time.sleep(SOCKET_WAIT_TIME)
            camera_thread = None

def process_files(input_path: Union[str, List[str]], 
                  extensions: List[str], 
                  F: Callable[[str], None]) -> None:
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]

    def process_file(file_path: str):
        _, ext = os.path.splitext(file_path)
        if ext.lower() in extensions:
            F(file_path)

    if isinstance(input_path, list):
        for file_path in input_path:
            process_file(file_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                process_file(os.path.join(root, file))
    else:
        raise ValueError("input_path must be a directory path or a list of file paths.")

def F(file_path):
    global filenames
    logger.info(f"Processing file: {file_path}")
    filenames.append(file_path)

def main():
    global camera_name
    global openai_model
    global openai_size
    global openai_quality
    global openai_style

    # Initialize the image to send with something
    initial_image = Image.new("L", (YAIL_W,YAIL_H))
    update_yail_data(pack_shades(initial_image), GRAPHICS_8)

    # Default OpenAI parameters
    openai_model = 'dall-e-3'
    openai_size = '1024x1024'
    openai_quality = 'standard'
    openai_style = 'vivid'

    bind_ip = '0.0.0.0'
    bind_port = 5556

    # Check if any arguments were provided (other than the script name)
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Yeets images to YAIL")
        parser.add_argument('paths', nargs='?', default=None, help='Directory path or list of file paths')
        parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.gif', '.png'], help='List of file extensions to process', required=False)
        parser.add_argument('--camera', nargs='?', default=None, help='The camera device to use', required=False)
        parser.add_argument('--port', nargs='+', default=None, help='Specify the port to listen too', required=False)
        parser.add_argument('--loglevel', nargs='+', default=None, help='The level of logging', required=False)
        parser.add_argument('--openai-api-key', type=str, help='OpenAI API key for image generation', required=False)
        parser.add_argument('--openai-model', type=str, default='dall-e-3', choices=['dall-e-3', 'gpt-4o'], help='OpenAI image generation model to use', required=False)
        parser.add_argument('--openai-size', type=str, default='1024x1024', choices=['1024x1024', '1792x1024', '1024x1792'], help='Image size for DALL-E 3 (ignored for GPT-4o)', required=False)
        parser.add_argument('--openai-quality', type=str, default='standard', choices=['standard', 'hd'], help='Image quality for DALL-E 3 (ignored for GPT-4o)', required=False)
        parser.add_argument('--openai-style', type=str, default='vivid', choices=['vivid', 'natural'], help='Image style for DALL-E 3 (ignored for GPT-4o)', required=False)
        
        args = parser.parse_args()

        if args.camera:
            camera_name = args.camera
        
        if args.openai_api_key:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key
        
        if args.openai_model:
            openai_model = args.openai_model
            
        if args.openai_size:
            openai_size = args.openai_size
            
        if args.openai_quality:
            openai_quality = args.openai_quality
            
        if args.openai_style:
            openai_style = args.openai_style
        
        if args.paths is not None and len(args.paths) == 1 and os.path.isdir(args.paths[0]):
            # If a single argument is passed and it's a directory
            directory_path = args.paths[0]
            logger.info("Processing files in directory:")
            process_files(directory_path, args.extensions, F)
        elif args.paths:
            # If multiple file paths are passed
            file_list = args.paths
            logger.info("Processing specific files in list:")
            process_files(file_list, args.extensions, F)

        if args.loglevel:
            loglevel = args.loglevel[0].upper()
            if loglevel == 'DEBUG':
                logger.setLevel(logging.DEBUG)
            elif loglevel == 'INFO':
                logger.setLevel(logging.INFO)
            elif loglevel == 'WARN':
                logger.setLevel(logging.WARN)
            elif loglevel == 'ERROR':
                logger.setLevel(logging.ERROR)
            elif loglevel == 'CRITICAL':
                logger.setLevel(logging.CRITICAL)

        if args.port:
            bind_port = int(args.port[0])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #server.bind((bind_ip, bind_port))
    server.bind(('', bind_port))
    server.listen(10)  # max backlog of connections

    #logger.info('Listening on {}:{}'.format(bind_ip, bind_port))
    logger.info('Listening on {}:{}'.format('', bind_port))

    while True:
        client_sock, address = server.accept()
        logger.info('Accepted connection from {}:{}'.format(address[0], address[1]))
        client_handler = Thread(
            target=handle_client_connection,
            args=(client_sock,)  # without comma you'd get a... TypeError: handle_client_connection() argument after * must be a sequence, not _socketobject
        )
        client_handler.daemon = True
        client_handler.start()

if __name__ == "__main__":
    main()
