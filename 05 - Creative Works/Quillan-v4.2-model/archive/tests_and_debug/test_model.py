#!/usr/bin/env python3
"""
Simple test script to check if the model can be imported and initialized
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test if we can import the model components"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import importlib.util
        
        # Load the main model file
        spec = importlib.util.spec_from_file_location("quillan_main", "🧠 Quillan v4.py")
        quillan_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quillan_main)
        
        print("[OK] Successfully loaded Quillan main module")
        
        # Check if classes exist
        assert hasattr(quillan_main, 'QuillanRoninV5_3'), "QuillanRoninV5_3 class not found"
        assert hasattr(quillan_main, 'Config'), "Config class not found"
        
        print("[OK] Found QuillanRoninV5_3 and Config classes")
        
        # Test config creation
        config = quillan_main.Config()
        print(f"[OK] Created config with hidden_dim={config.hidden_dim}, num_experts={config.num_experts}")
        
        return True, quillan_main
        
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_initialization(quillan_main):
    """Test if we can initialize the model (without torch for now)"""
    try:
        print("\nTesting model structure...")
        
        # Check if we can at least create the class without torch
        model_class = quillan_main.QuillanRoninV5_3
        
        # Check the forward method signature
        import inspect
        sig = inspect.signature(model_class.forward)
        print(f"[OK] Model forward method signature: {sig}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Model structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Quillan Model Test ===\n")
    
    # Test imports
    success, quillan_main = test_import()
    if not success:
        print("\n[ERROR] Cannot proceed without successful imports")
        return
    
    # Test model structure
    success = test_model_initialization(quillan_main)
    if not success:
        print("\n[ERROR] Model structure test failed")
        return
    
    print("\n[SUCCESS] All basic tests passed!")
    print("\nNext steps:")
    print("1. Install PyTorch: pip install torch")
    print("2. Run: python train.py")
    print("3. Or run: python inference.py")

if __name__ == "__main__":
    main()
