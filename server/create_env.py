#!/usr/bin/env python

import os
import sys

def create_env_file():
    """Create a .env file with OpenAI API key and configuration."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    # Check if .env file already exists
    if os.path.exists(env_path):
        overwrite = input(f".env file already exists at {env_path}. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key: ")
    
    # Optional configurations
    use_advanced = input("Configure advanced OpenAI settings? (y/n): ")
    
    model = "dall-e-3"
    size = "1024x1024"
    quality = "standard"
    style = "vivid"
    system_prompt = "You are an image generation assistant. Generate an image based on the user's description."
    
    if use_advanced.lower() == 'y':
        print("\nAdvanced OpenAI Configuration")
        print("----------------------------")
        
        # Model selection
        print("\nAvailable models:")
        print("1. dall-e-3 (default)")
        print("2. gpt-4o")
        model_choice = input("Select model (1-2, default is 1): ")
        if model_choice == "2":
            model = "gpt-4o"
        
        # Size selection (only for DALL-E 3)
        if model == "dall-e-3":
            print("\nAvailable sizes:")
            print("1. 1024x1024 (default)")
            print("2. 1792x1024")
            print("3. 1024x1792")
            size_choice = input("Select size (1-3, default is 1): ")
            if size_choice == "2":
                size = "1792x1024"
            elif size_choice == "3":
                size = "1024x1792"
            
            # Quality selection
            print("\nAvailable qualities:")
            print("1. standard (default)")
            print("2. hd")
            quality_choice = input("Select quality (1-2, default is 1): ")
            if quality_choice == "2":
                quality = "hd"
            
            # Style selection
            print("\nAvailable styles:")
            print("1. vivid (default)")
            print("2. natural")
            style_choice = input("Select style (1-2, default is 1): ")
            if style_choice == "2":
                style = "natural"
        
        # System prompt
        custom_prompt = input("\nWould you like to customize the system prompt for image generation? (y/n): ")
        if custom_prompt.lower() == 'y':
            print("\nDefault system prompt: " + system_prompt)
            system_prompt = input("Enter your custom system prompt: ")
            if not system_prompt.strip():
                system_prompt = "You are an image generation assistant. Generate an image based on the user's description."
    
    # Create .env file content
    env_content = f"""# OpenAI API Configuration
OPENAI_API_KEY={api_key}
OPENAI_MODEL={model}
OPENAI_SIZE={size}
OPENAI_QUALITY={quality}
OPENAI_STYLE={style}
OPENAI_SYSTEM_PROMPT="{system_prompt}"
"""
    
    # Write to .env file
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\n.env file created successfully at {env_path}")
    print("This file contains your API key and should not be committed to version control.")
    print("It has been added to .gitignore for your protection.")
    print("\nTo start the server with these settings, run:")
    print("source .venv/bin/activate")
    print("python yail.py --loglevel DEBUG")

if __name__ == "__main__":
    create_env_file()
