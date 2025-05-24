#!/usr/bin/env python3
"""
Verification script for OpenClassifier installation.
Tests basic imports and functionality.
"""

import sys
import traceback

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")
    
    try:
        # Test core imports
        from open_classifier.core.config import settings
        print("✓ Core config imported successfully")
        
        from open_classifier.core.logging import get_logger
        print("✓ Core logging imported successfully")
        
        from open_classifier.core.exceptions import ClassifierError
        print("✓ Core exceptions imported successfully")
        
        # Test utility imports
        from open_classifier.utils.text_utils import clean_text
        print("✓ Text utilities imported successfully")
        
        from open_classifier.utils.cache import LRUCache
        print("✓ Cache utilities imported successfully")
        
        from open_classifier.utils.data_utils import DataLoader
        print("✓ Data utilities imported successfully")
        
        # Test API models
        from open_classifier.api.models import ClassificationRequest
        print("✓ API models imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test text cleaning
        from open_classifier.utils.text_utils import clean_text
        cleaned = clean_text("  Hello World!  ")
        assert cleaned == "Hello World!", f"Expected 'Hello World!', got '{cleaned}'"
        print("✓ Text cleaning works correctly")
        
        # Test cache
        from open_classifier.utils.cache import LRUCache
        cache = LRUCache(max_size=10)
        cache.set("test", "value")
        assert cache.get("test") == "value", "Cache get/set failed"
        print("✓ LRU cache works correctly")
        
        # Test data loader
        from open_classifier.utils.data_utils import DataLoader
        loader = DataLoader()
        assert loader.batch_size == 1000, "DataLoader initialization failed"
        print("✓ DataLoader initialization works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from open_classifier.core.config import settings
        
        # Test that settings object exists and has expected attributes
        assert hasattr(settings, 'app_name'), "Missing app_name in settings"
        assert hasattr(settings, 'debug'), "Missing debug in settings"
        print("✓ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("OpenClassifier Installation Verification")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! OpenClassifier is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 