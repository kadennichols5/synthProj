import sys
import os
import pytest
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spectral_analysis import Byte, ByteExtractor



class TestByte:
    @pytest.fixture
    def sample_byte(self):
        """Create a sample Byte for testing"""
        return Byte(
            time=0.0,
            stem_label='vocals',
            magnitudes=np.random.rand(1024),
            phases=np.random.uniform(-np.pi, np.pi, 1024),
            frequency_bins=np.linspace(0, 22050, 1024),
            sample_rate=44100
        )

    def test_byte_initialization(self, sample_byte):
        """Test if Byte initializes correctly"""
        assert isinstance(sample_byte.magnitudes, np.ndarray)
        assert isinstance(sample_byte.phases, np.ndarray)
        assert len(sample_byte.magnitudes) == len(sample_byte.phases)

    def test_get_vector(self, sample_byte):
        """Test if get_vector returns normalized values"""
        vector = sample_byte.get_vector()
        assert len(vector) == len(sample_byte.magnitudes) * 2
        assert np.all(vector >= 0) and np.all(vector <= 1)

    def test_frequency_bands(self, sample_byte):
        """Test frequency band analysis"""
        bands = sample_byte.get_frequency_bands()
        assert all(band in bands for band in 
                  ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high'])
        assert all(energy >= 0 for energy in bands.values())

class TestByteExtractor:
    @pytest.fixture
    def extractor(self):
        return ByteExtractor()

    def test_extract_bytes(self, extractor):
        """Test byte extraction from synthetic audio"""
        # Create synthetic audio (1 second of 440Hz sine wave)
        duration = 1.0
        t = np.linspace(0, duration, int(44100 * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        bytes_list = extractor.extract_bytes(audio, 'test')
        assert len(bytes_list) > 0
        assert all(isinstance(b, Byte) for b in bytes_list)
