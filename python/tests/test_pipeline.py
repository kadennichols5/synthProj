#!/usr/bin/env python3
"""
Basic tests for the audio processing pipeline components.

This script verifies that all major components can be imported
and basic functionality works correctly.
"""

import os
import sys
import tempfile
import numpy as np
import unittest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestPipelineComponents(unittest.TestCase):
    """Test basic functionality of pipeline components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_imports(self):
        """Test that all components can be imported."""
        try:
            from segment_generator import extract_segments
            from audio_separation import AudioSeparator
            from spectral_analysis import ByteExtractor, Byte
            from process_stems import process_stem_file
            from neural_network import VisualEncoder
            print("✅ All components imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import component: {e}")
    
    def test_byte_creation(self):
        """Test Byte object creation and validation."""
        from spectral_analysis import Byte
        
        # Create test data
        time = 1.0
        stem_label = "vocals"
        magnitudes = np.random.rand(1024).astype(np.float32)
        phases = np.random.rand(1024).astype(np.float32) * 2 * np.pi - np.pi
        frequency_bins = np.linspace(0, 22050, 1024).astype(np.float32)
        sample_rate = 44100
        
        # Create Byte object
        byte = Byte(
            time=time,
            stem_label=stem_label,
            magnitudes=magnitudes,
            phases=phases,
            frequency_bins=frequency_bins,
            sample_rate=sample_rate
        )
        
        # Test properties
        self.assertEqual(byte.time, time)
        self.assertEqual(byte.stem_label, stem_label)
        self.assertEqual(byte.sample_rate, sample_rate)
        self.assertEqual(len(byte.magnitudes), 1024)
        self.assertEqual(len(byte.phases), 1024)
        self.assertEqual(len(byte.frequency_bins), 1024)
        
        # Test vector generation
        vector = byte.get_vector()
        self.assertEqual(len(vector), 2048)  # magnitudes + phases
        self.assertTrue(np.all(vector >= 0) and np.all(vector <= 1))  # normalized
        
        print("✅ Byte object creation and validation passed")
    
    def test_byte_extractor_initialization(self):
        """Test ByteExtractor initialization."""
        from spectral_analysis import ByteExtractor
        
        # Test default initialization
        extractor = ByteExtractor()
        self.assertEqual(extractor.window_size, 2048)
        self.assertEqual(extractor.hop_length, 512)
        self.assertEqual(extractor.sample_rate, 44100)
        
        # Test custom initialization
        extractor = ByteExtractor(
            window_size=1024,
            hop_length=256,
            sample_rate=22050
        )
        self.assertEqual(extractor.window_size, 1024)
        self.assertEqual(extractor.hop_length, 256)
        self.assertEqual(extractor.sample_rate, 22050)
        
        print("✅ ByteExtractor initialization passed")
    
    def test_audio_separator_initialization(self):
        """Test AudioSeparator initialization."""
        from audio_separation import AudioSeparator
        
        input_path = "test_input.wav"
        output_path = self.temp_dir
        
        separator = AudioSeparator(input_path, output_path)
        
        self.assertEqual(separator.input_path, input_path)
        self.assertEqual(separator.output_path, output_path)
        self.assertEqual(separator.model_name, 'htdemucs')
        self.assertTrue(separator.mp3)
        
        print("✅ AudioSeparator initialization passed")
    
    def test_neural_network_initialization(self):
        """Test VisualEncoder initialization."""
        from neural_network import VisualEncoder
        
        # Test initialization
        input_dim = 4096  # 2048 magnitudes + 2048 phases
        encoder = VisualEncoder(input_dim=input_dim)
        
        # Test forward pass with dummy data
        dummy_input = np.random.randn(1, input_dim).astype(np.float32)
        import torch
        dummy_tensor = torch.tensor(dummy_input)
        
        try:
            outputs = encoder(dummy_tensor)
            
            # Check output structure
            expected_keys = ['shape', 'motion', 'texture', 'color', 'brightness', 'position', 'pattern']
            for key in expected_keys:
                self.assertIn(key, outputs)
                self.assertIsInstance(outputs[key], torch.Tensor)
            
            print("✅ VisualEncoder initialization and forward pass passed")
            
        except Exception as e:
            self.fail(f"VisualEncoder forward pass failed: {e}")
    
    def test_path_configuration(self):
        """Test path configuration system."""
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
        
        try:
            from paths import get_segmented_audio_path, get_segmented_stems_path, get_spectral_data_path
            
            # Test that paths are returned as strings
            audio_path = get_segmented_audio_path()
            stems_path = get_segmented_stems_path()
            spectral_path = get_spectral_data_path()
            
            self.assertIsInstance(audio_path, str)
            self.assertIsInstance(stems_path, str)
            self.assertIsInstance(spectral_path, str)
            
            print("✅ Path configuration system passed")
            
        except ImportError as e:
            self.fail(f"Failed to import path configuration: {e}")
    
    def test_spectral_analysis_with_dummy_data(self):
        """Test spectral analysis with dummy audio data."""
        from spectral_analysis import ByteExtractor
        
        # Create dummy audio data (1 second of random audio)
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples).astype(np.float32)
        
        # Initialize extractor
        extractor = ByteExtractor(sample_rate=sample_rate)
        
        # Extract bytes
        bytes_list = extractor.extract_bytes(audio_data, "test_stem")
        
        # Verify results
        self.assertGreater(len(bytes_list), 0)
        
        for byte in bytes_list:
            self.assertEqual(byte.stem_label, "test_stem")
            self.assertEqual(byte.sample_rate, sample_rate)
            self.assertGreaterEqual(byte.time, 0)
            self.assertLess(byte.time, duration)
            
            # Check that magnitudes and phases have same length
            self.assertEqual(len(byte.magnitudes), len(byte.phases))
            self.assertEqual(len(byte.magnitudes), len(byte.frequency_bins))
        
        print("✅ Spectral analysis with dummy data passed")

def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Running Audio Processing Pipeline Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPipelineComponents)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 50)
    if result.wasSuccessful():
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1) 