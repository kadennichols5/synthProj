#!/usr/bin/env python3
"""
Setup script for the Audio Processing Pipeline.

This script helps set up the environment and verify that all
components are working correctly.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def install_demucs():
    """Install Demucs for audio separation."""
    print("\n🎼 Installing Demucs...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "demucs"
        ])
        print("✅ Demucs installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Demucs: {e}")
        return False

def check_imports():
    """Check if all required modules can be imported."""
    print("\n🔍 Checking imports...")
    
    required_modules = [
        'numpy',
        'scipy',
        'soundfile',
        'pydub',
        'torch',
        'torchaudio',
        'demucs'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All required modules imported successfully")
    return True

def check_pipeline_components():
    """Check if pipeline components can be imported."""
    print("\n🔧 Checking pipeline components...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    pipeline_modules = [
        'segment_generator',
        'audio_separation', 
        'spectral_analysis',
        'process_stems',
        'neural_network'
    ]
    
    failed_imports = []
    
    for module in pipeline_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import pipeline components: {', '.join(failed_imports)}")
        return False
    
    print("✅ All pipeline components imported successfully")
    return True

def check_paths():
    """Check if path configuration works."""
    print("\n📁 Checking path configuration...")
    
    try:
        # Add config to path
        config_path = Path(__file__).parent / "config"
        sys.path.insert(0, str(config_path))
        
        from paths import get_segmented_audio_path, get_segmented_stems_path, get_spectral_data_path
        
        # Test path functions
        audio_path = get_segmented_audio_path()
        stems_path = get_segmented_stems_path()
        spectral_path = get_spectral_data_path()
        
        print(f"✅ Segmented audio path: {audio_path}")
        print(f"✅ Segmented stems path: {stems_path}")
        print(f"✅ Spectral data path: {spectral_path}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import path configuration: {e}")
        return False

def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Run test script
        test_script = Path(__file__).parent / "tests" / "test_pipeline.py"
        if test_script.exists():
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Basic tests passed")
                return True
            else:
                print("❌ Basic tests failed")
                print(result.stdout)
                print(result.stderr)
                return False
        else:
            print("⚠️  Test script not found, skipping tests")
            return True
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📂 Creating directories...")
    
    try:
        # Add config to path
        config_path = Path(__file__).parent / "config"
        sys.path.insert(0, str(config_path))
        
        from paths import get_segmented_audio_path, get_segmented_stems_path, get_spectral_data_path
        
        directories = [
            get_segmented_audio_path(),
            get_segmented_stems_path(),
            get_spectral_data_path()
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False

def main():
    """Main setup function."""
    print("🎵 Audio Processing Pipeline Setup")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Install Requirements", install_requirements),
        ("Install Demucs", install_demucs),
        ("Check Imports", check_imports),
        ("Check Pipeline Components", check_pipeline_components),
        ("Check Paths", check_paths),
        ("Create Directories", create_directories),
        ("Run Basic Tests", run_basic_tests)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}...")
        if not check_func():
            failed_checks.append(check_name)
    
    print("\n" + "=" * 40)
    if not failed_checks:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your audio files to the BassVocalDrumGuitar directory")
        print("2. Run: python src/main.py --mode full")
        print("3. Or try the demo: python examples/basic_pipeline_demo.py <audio_file>")
        return True
    else:
        print("❌ Setup failed!")
        print(f"Failed checks: {', '.join(failed_checks)}")
        print("\nPlease fix the issues above and run setup again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 