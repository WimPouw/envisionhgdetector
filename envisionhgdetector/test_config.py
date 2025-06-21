#!/usr/bin/env python3
"""
Test script for the current EnvisionHGDetector setup.
This works with your existing installed package and shows what needs to be updated.
"""

import os
import sys

def test_current_setup():
    """Test the current installation and identify what needs updating."""
    try:
        from envisionhgdetector import Config, GestureDetector
        
        print("=" * 60)
        print("EnvisionHGDetector Current Setup Analysis")
        print("=" * 60)
        
        # Test basic imports that should work
        print("\n1. Testing current imports...")
        try:
            from envisionhgdetector import GestureModel, make_model
            print("   ✓ GestureModel and make_model imported successfully")
        except ImportError as e:
            print(f"   ✗ Original model imports failed: {e}")
        
        # Test config
        print("\n2. Testing current Config...")
        config = Config()
        print(f"   ✓ Config initialized: {config}")
        
        # Show current config attributes
        print(f"   - Gesture labels: {config.gesture_labels}")
        print(f"   - Sequence length: {config.seq_length}")
        if hasattr(config, 'weights_path'):
            print(f"   - Weights path: {config.weights_path}")
            if config.weights_path and os.path.exists(config.weights_path):
                print("   ✓ CNN model file exists")
            else:
                print("   ✗ CNN model file not found")
        
        # Test detector
        print("\n3. Testing current GestureDetector...")
        try:
            detector = GestureDetector()
            print("   ✓ GestureDetector initialized successfully")
            print(f"   - Model type: {getattr(detector, 'model_type', 'cnn (default)')}")
        except Exception as e:
            print(f"   ✗ GestureDetector failed: {e}")
        
        # Check for new features
        print("\n4. Checking for new features...")
        
        # Check if LightGBM support exists
        try:
            detector_with_type = GestureDetector(model_type="lightgbm")
            print("   ✓ LightGBM support already available")
        except Exception as e:
            print(f"   ✗ LightGBM support not available: {e}")
        
        # Check if RealtimeGestureDetector exists
        try:
            from envisionhgdetector import RealtimeGestureDetector
            print("   ✓ RealtimeGestureDetector already available")
        except ImportError:
            print("   ✗ RealtimeGestureDetector not available - needs update")
        
        # Check package location
        import envisionhgdetector
        package_path = os.path.dirname(envisionhgdetector.__file__)
        print(f"\n5. Package location: {package_path}")
        
        # List files in package
        print("\n6. Current package files:")
        try:
            files = os.listdir(package_path)
            for file in sorted(files):
                if file.endswith(('.py', '.h5', '.pkl')):
                    print(f"   - {file}")
        except Exception as e:
            print(f"   Error listing files: {e}")
        
        # Check for model files
        print("\n7. Looking for model files...")
        model_dir = os.path.join(package_path, 'model')
        if os.path.exists(model_dir):
            print(f"   Model directory found: {model_dir}")
            try:
                model_files = os.listdir(model_dir)
                for file in sorted(model_files):
                    print(f"   - {file}")
            except Exception as e:
                print(f"   Error listing model files: {e}")
        else:
            print("   ✗ Model directory not found")
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def show_update_instructions():
    """Show instructions for updating the package."""
    print("\n" + "=" * 60)
    print("UPDATE INSTRUCTIONS")
    print("=" * 60)
    
    print("\nTo add LightGBM and real-time detection support:")
    print("\n1. Update your package files with the new versions I provided:")
    print("   - config.py (enhanced with LightGBM support)")
    print("   - __init__.py (updated imports)")
    print("   - Add model_lightgbm.py to your package")
    print("   - Update detector.py with LightGBM integration")
    
    print("\n2. Add your LightGBM model file:")
    print("   - Place lightgbm_gesture_model_v1.pkl in the model/ directory")
    
    print("\n3. Reinstall the package:")
    print("   pip uninstall envisionhgdetector")
    print("   pip install . (from your package source directory)")
    
    print("\n4. Or if developing locally:")
    print("   pip install -e . (editable install)")
    
    import envisionhgdetector
    package_path = os.path.dirname(envisionhgdetector.__file__)
    print(f"\nYour package files are located at:")
    print(f"{package_path}")

def create_development_test():
    """Create a test that can work with development files."""
    print("\n" + "=" * 60)
    print("DEVELOPMENT TEST OPTION")
    print("=" * 60)
    
    print("\nIf you want to test without reinstalling:")
    print("1. Copy the updated files to your development directory")
    print("2. Run Python from that directory")
    print("3. The import will use local files instead of installed package")
    
    test_code = '''
# test_local_development.py
import sys
import os

# Add current directory to path to use local files
sys.path.insert(0, '.')

try:
    from config import Config
    from detector import GestureDetector, RealtimeGestureDetector
    
    print("✓ Local imports successful!")
    
    # Test new config features
    config = Config()
    print(f"Config: {config}")
    
    # Test model availability
    if hasattr(config, 'validate_model_availability'):
        cnn_available = config.validate_model_availability('cnn')
        lightgbm_available = config.validate_model_availability('lightgbm')
        print(f"CNN available: {cnn_available}")
        print(f"LightGBM available: {lightgbm_available}")
    
except ImportError as e:
    print(f"Import error: {e}")
'''
    
    print("\nSample local development test code:")
    print(test_code)

def main():
    """Run the current setup analysis."""
    print("Analyzing your current EnvisionHGDetector setup...\n")
    
    success = test_current_setup()
    show_update_instructions()
    create_development_test()
    
    if success:
        print("\n🔍 Analysis completed! See instructions above for updating.")
    else:
        print("\n❌ Analysis failed. Please check your installation.")

if __name__ == "__main__":
    main()