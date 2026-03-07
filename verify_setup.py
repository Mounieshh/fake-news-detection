#!/usr/bin/env python3
"""
Verification script to test API configuration and connectivity
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_variables():
    """Check if required environment variables are set"""
    print("=" * 60)
    print("Checking Environment Variables...")
    print("=" * 60)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        print("✓ GEMINI_API_KEY is configured")
        return True
    else:
        print("✗ GEMINI_API_KEY is NOT configured")
        print("  Please add your Gemini API key to .env file")
        return False

def check_imports():
    """Check if all required modules can be imported"""
    print("\n" + "=" * 60)
    print("Checking Required Imports...")
    print("=" * 60)
    
    modules = [
        "flask",
        "torch",
        "torchvision",
        "requests",
        "dotenv",
        "google",
        "PIL"
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_src_modules():
    """Check if src modules can be imported"""
    print("\n" + "=" * 60)
    print("Checking Project Modules...")
    print("=" * 60)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.image_predict.image_model import ImageFakeNewsModel
        print("✓ ImageFakeNewsModel")
        
        # Try to initialize the model
        try:
            model = ImageFakeNewsModel()
            print("✓ Image model backend initialized successfully")
            
            # Test API connection
            if model.is_loaded and model.backend:
                print("\nTesting API connection...")
                result = model.backend.test_connection()
                if result["status"] == "SUCCESS":
                    print(f"✓ API connection successful: {result['message']}")
                    return True
                else:
                    print(f"✗ API connection failed: {result['message']}")
                    return False
            else:
                print(f"✗ Model not loaded: {model.error_message}")
                return False
        except ValueError as e:
            print(f"✗ Image model initialization failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Failed to import image model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTION - API CONFIGURATION VERIFICATION")
    print("=" * 60 + "\n")
    
    env_ok = check_env_variables()
    imports_ok = check_imports()
    modules_ok = check_src_modules()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if env_ok and imports_ok and modules_ok:
        print("✓ All checks passed! Your API is configured correctly.")
        print("\nYou can now run the application with:")
        print("  python main.py")
        return 0
    else:
        print("✗ Some checks failed. Please review the issues above.")
        print("\nFor help, see API_SETUP.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
