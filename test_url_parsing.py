#!/usr/bin/env python3
"""
Test script to verify URL parsing functionality for the image classifier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.services.image_classifier import ImageClassifier

def test_url_parsing():
    classifier = ImageClassifier()
    
    # Test case 1: Simple URL
    simple_url = "https://assets.wfcdn.com/im/55507952/resize-h300-w300%5Ecompr-r85/sofa.jpg"
    print("Test 1 - Simple URL:")
    print(f"Input: {simple_url}")
    result = classifier.parse_image_url(simple_url)
    print(f"Output: {result}")
    print()
    
    # Test case 2: Srcset format (the problematic one from your logs)
    srcset_url = "https://assets.wfcdn.com/im/55507952/resize-h300-w300%5Ecompr-r85/2477/247755988/Winfree+78.8%27%27+Upholstered+Sofa.jpg 300w, https://assets.wfcdn.com/im/39504752/resize-h400-w400%5Ecompr-r85/2477/247755988/Winfree+78.8%27%27+Upholstered+Sofa.jpg 400w, https://assets.wfcdn.com/im/60449840/resize-h500-w500%5Ecompr-r85/2477/247755988/Winfree+78.8%27%27+Upholstered+Sofa.jpg 500w, https://assets.wfcdn.com/im/34562864/resize-h600-w600%5Ecompr-r85/2477/247755988/Winfree+78.8%27%27+Upholstered+Sofa.jpg 600w, https://assets.wfcdn.com/im/29575568/resize-h700-w700%5Ecompr-r85/2477/247755988/Winfree+78.8%27%27+Upholstered+Sofa.jpg 700w, https://assets.wfcdn.com/im/76362864/resize-h755-w755%5Ecompr-r85/2477/247755988/Winfree+78.8%27%27+Upholstered+Sofa.jpg 755w, https://assets.wfcdn.com/im/70379024/resize-h800-w800%5Ecompr-r85/2477/247755988/Winfree+78.8%27%27+Upholstered+Sofa.jpg 800w"
    print("Test 2 - Srcset format (multiple URLs with sizes):")
    print(f"Input: {srcset_url[:100]}...")
    result = classifier.parse_image_url(srcset_url)
    print(f"Output: {result}")
    print()
    
    # Test case 3: Empty/None input
    print("Test 3 - Empty/None input:")
    print(f"Input: None")
    result = classifier.parse_image_url(None)
    print(f"Output: {result}")
    print()
    
    print("Test 4 - Empty string:")
    print(f"Input: ''")
    result = classifier.parse_image_url("")
    print(f"Output: {result}")
    print()
    
    # Test case 5: Invalid input
    print("Test 5 - Invalid input:")
    invalid_url = "not a url"
    print(f"Input: {invalid_url}")
    result = classifier.parse_image_url(invalid_url)
    print(f"Output: {result}")
    print()

if __name__ == "__main__":
    print("Testing URL parsing functionality...")
    print("=" * 50)
    test_url_parsing()
    print("=" * 50)
    print("Testing complete!")