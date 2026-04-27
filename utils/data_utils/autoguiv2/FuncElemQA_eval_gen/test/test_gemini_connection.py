#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Gemini API connection and error handling.

This script tests:
1. Basic API connection
2. Image-based API calls (simulating the main script usage)
3. Retry mechanism with exponential backoff
4. Detailed error reporting

Usage:
    python test_gemini_connection.py \
        --base-url "https://xiaoai.plus/v1" \
        --api-key "your_api_key" \
        --model "gemini-2.5-pro-thinking" \
        [--image-path "path/to/test/image.png"] \
        [--max-retries 5] \
        [--debug]
"""

import os
import sys
import time
import argparse
import base64
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.openai_utils.openai import OpenAIModel

# Colorized output support
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:
    class _Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = WHITE = ""
    class _Style:
        RESET_ALL = ""
    Fore = _Fore()
    Style = _Style()


def print_colored(message: str, level: str = "info"):
    """Print colored message"""
    color_map = {
        "error": Fore.RED,
        "success": Fore.GREEN,
        "warn": Fore.YELLOW,
        "info": Fore.CYAN,
        "step": Fore.MAGENTA,
    }
    prefix_map = {
        "error": "❌ ERROR",
        "success": "✅ SUCCESS",
        "warn": "⚠️  WARN",
        "info": "ℹ️  INFO",
        "step": "🔹 STEP",
    }
    color_code = color_map.get(level, Fore.WHITE)
    prefix = prefix_map.get(level, "")
    print(f"{color_code}{prefix}: {message}{Style.RESET_ALL}")


def image_to_base64(image_path: str) -> str:
    """Convert an image to a base64 data URL."""
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    
    ext = Path(image_path).suffix.lower()
    with open(image_path, "rb") as f:
        binary_data = f.read()
    base64_data = base64.b64encode(binary_data).decode("utf-8")
    return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"


def test_basic_connection(model: OpenAIModel, debug: bool = False) -> Dict[str, Any]:
    """Test basic text-only API connection"""
    print_colored("=" * 60, "step")
    print_colored("Test 1: Basic Text-Only Connection", "step")
    print_colored("=" * 60, "step")
    
    test_prompt = "Please respond with a simple JSON: {\"status\": \"ok\", \"message\": \"connection successful\"}"
    
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': test_prompt}
        ]
    }]
    
    start_time = time.time()
    try:
        success, response, _ = model.get_model_response_with_prepared_messages(
            messages, temperature=0.2, timeout=60
        )
        elapsed = time.time() - start_time
        
        if success:
            print_colored(f"✅ Connection successful! ({elapsed:.2f}s)", "success")
            print_colored(f"Response: {response[:200]}...", "info")
            return {"success": True, "elapsed": elapsed, "response": response}
        else:
            print_colored(f"❌ API call failed: {response}", "error")
            return {"success": False, "elapsed": elapsed, "error": response}
            
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)
        print_colored(f"❌ Exception occurred ({elapsed:.2f}s):", "error")
        print_colored(f"   Type: {error_type}", "error")
        print_colored(f"   Message: {error_msg}", "error")
        
        if debug:
            import traceback
            print_colored("   Full traceback:", "error")
            traceback.print_exc()
        
        return {"success": False, "elapsed": elapsed, "error_type": error_type, "error": error_msg}


def test_image_connection(model: OpenAIModel, image_path: str, debug: bool = False) -> Dict[str, Any]:
    """Test image-based API connection (simulating main script usage)"""
    print_colored("=" * 60, "step")
    print_colored("Test 2: Image-Based Connection", "step")
    print_colored("=" * 60, "step")
    
    if not os.path.exists(image_path):
        print_colored(f"❌ Image not found: {image_path}", "error")
        return {"success": False, "error": "Image file not found"}
    
    try:
        image_base64 = image_to_base64(image_path)
        print_colored(f"✅ Image loaded: {image_path}", "success")
    except Exception as e:
        print_colored(f"❌ Failed to load image: {e}", "error")
        return {"success": False, "error": str(e)}
    
    # Simulate a visual verification prompt (similar to main script)
    prompt = """You are a GUI understanding expert. Please analyze this image and respond with a JSON:
{
    "status": "ok",
    "image_analyzed": true,
    "message": "Image successfully processed"
}"""
    
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': image_base64}},
            {'type': 'text', 'text': prompt}
        ]
    }]
    
    start_time = time.time()
    try:
        success, response, _ = model.get_model_response_with_prepared_messages(
            messages, temperature=0.2, timeout=300
        )
        elapsed = time.time() - start_time
        
        if success:
            print_colored(f"✅ Image API call successful! ({elapsed:.2f}s)", "success")
            print_colored(f"Response: {response[:200]}...", "info")
            return {"success": True, "elapsed": elapsed, "response": response}
        else:
            print_colored(f"❌ Image API call failed: {response}", "error")
            return {"success": False, "elapsed": elapsed, "error": response}
            
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)
        print_colored(f"❌ Exception occurred ({elapsed:.2f}s):", "error")
        print_colored(f"   Type: {error_type}", "error")
        print_colored(f"   Message: {error_msg}", "error")
        
        if debug:
            import traceback
            print_colored("   Full traceback:", "error")
            traceback.print_exc()
        
        return {"success": False, "elapsed": elapsed, "error_type": error_type, "error": error_msg}


def test_with_retry(model: OpenAIModel, max_retries: int = 5, debug: bool = False) -> Dict[str, Any]:
    """Test API connection with retry mechanism (simulating main script behavior)"""
    print_colored("=" * 60, "step")
    print_colored(f"Test 3: Connection with Retry Mechanism (max {max_retries} retries)", "step")
    print_colored("=" * 60, "step")
    
    test_prompt = "Please respond with: OK"
    
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': test_prompt}
        ]
    }]
    
    for attempt in range(max_retries):
        try:
            if debug:
                print_colored(f"   Attempt {attempt + 1}/{max_retries}", "info")
            
            # Add exponential backoff delay for retries
            if attempt > 0:
                delay = min(2 ** attempt, 30)  # Max 30 seconds
                if debug:
                    print_colored(f"   Waiting {delay}s before retry...", "info")
                time.sleep(delay)
            
            start_time = time.time()
            success, response, _ = model.get_model_response_with_prepared_messages(
                messages, temperature=0.2 if attempt == 0 else 0.4, timeout=300
            )
            elapsed = time.time() - start_time
            
            if success:
                print_colored(f"✅ Success on attempt {attempt + 1}! ({elapsed:.2f}s)", "success")
                return {"success": True, "attempt": attempt + 1, "elapsed": elapsed, "response": response}
            else:
                if debug:
                    print_colored(f"   API call failed: {response}", "warn")
                continue
                
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if debug:
                print_colored(f"   Exception on attempt {attempt + 1}: {error_type}: {error_msg}", "warn")
            
            # If it's the last attempt, return the error
            if attempt == max_retries - 1:
                print_colored(f"❌ All {max_retries} attempts failed", "error")
                print_colored(f"   Last error: {error_type}: {error_msg}", "error")
                if debug:
                    import traceback
                    traceback.print_exc()
                return {"success": False, "attempts": max_retries, "error_type": error_type, "error": error_msg}
    
    return {"success": False, "attempts": max_retries, "error": "Unknown error"}


def main():
    parser = argparse.ArgumentParser(
        description="Test Gemini API connection and error handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python test_gemini_connection.py \\
      --base-url "https://xiaoai.plus/v1" \\
      --api-key "your_key" \\
      --model "gemini-2.5-pro-thinking"
  
  # Test with image
  python test_gemini_connection.py \\
      --base-url "https://xiaoai.plus/v1" \\
      --api-key "your_key" \\
      --model "gemini-2.5-pro-thinking" \\
      --image-path "test_image.png"
  
  # Test with retry mechanism
  python test_gemini_connection.py \\
      --base-url "https://xiaoai.plus/v1" \\
      --api-key "your_key" \\
      --model "gemini-2.5-pro-thinking" \\
      --max-retries 5 \\
      --debug
        """
    )
    
    parser.add_argument("--base-url", type=str, required=True,
                       help="API base URL (e.g., https://xiaoai.plus/v1)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-thinking",
                       help="Model name (default: gemini-2.5-pro-thinking)")
    parser.add_argument("--image-path", type=str, default=None,
                       help="Optional: Path to test image for image-based API test")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Maximum number of retries for retry test (default: 5)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output with full tracebacks")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_colored("❌ API key not provided!", "error")
        print_colored("   Please provide via --api-key or set OPENAI_API_KEY environment variable", "error")
        return 1
    
    # Initialize model
    print_colored("=" * 60, "step")
    print_colored("Initializing Gemini API Model", "step")
    print_colored("=" * 60, "step")
    print_colored(f"Base URL: {args.base_url}", "info")
    print_colored(f"Model: {args.model}", "info")
    print_colored(f"API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if len(api_key) > 8 else '***'}", "info")
    
    try:
        model = OpenAIModel(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            temperature=0.1,
            max_tokens=16000
        )
        print_colored("✅ Model initialized successfully", "success")
    except Exception as e:
        print_colored(f"❌ Failed to initialize model: {e}", "error")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    # Run tests
    results = {}
    
    # Test 1: Basic connection
    results["basic"] = test_basic_connection(model, debug=args.debug)
    print()
    
    # Test 2: Image connection (if image provided)
    if args.image_path:
        results["image"] = test_image_connection(model, args.image_path, debug=args.debug)
        print()
    
    # Test 3: Retry mechanism
    results["retry"] = test_with_retry(model, max_retries=args.max_retries, debug=args.debug)
    print()
    
    # Summary
    print_colored("=" * 60, "step")
    print_colored("Test Summary", "step")
    print_colored("=" * 60, "step")
    
    for test_name, result in results.items():
        if result.get("success"):
            print_colored(f"{test_name.upper()}: ✅ PASSED", "success")
        else:
            print_colored(f"{test_name.upper()}: ❌ FAILED", "error")
            if "error" in result:
                print_colored(f"   Error: {result['error']}", "error")
    
    # Return exit code
    all_passed = all(r.get("success", False) for r in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

