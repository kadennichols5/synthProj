"""
Centralized settings configuration for the Audio Processing Pipeline.
Handles environment variables, validation, and provides a clean interface.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AudioSettings:
    """Audio processing configuration settings."""
    sample_rate: int = 44100
    segment_length_ms: int = 5000
    stride_ms: int = 2000
    random_ratio: float = 0.3
    random_length_range: tuple = (2000, 5000)

@dataclass
class SpectralSettings:
    """Spectral analysis configuration settings."""
    window_size: int = 2048
    hop_length: int = 512
    n_fft: int = 2048
    frequency_bins: int = 2048

@dataclass
class ProcessingSettings:
    """Processing pipeline configuration settings."""
    batch_size: int = 4
    num_workers: int = 2
    use_gpu: bool = False
    max_memory_gb: float = 8.0

@dataclass
class DemucsSettings:
    """Demucs audio separation settings."""
    model_name: str = "htdemucs"
    shifts: int = 1
    split: bool = True
    overlap: float = 0.25

class Settings:
    """Main settings class that manages all configuration."""
    
    def __init__(self):
        # Create the dataclass instances first
        self.audio = AudioSettings()
        self.spectral = SpectralSettings()
        self.processing = ProcessingSettings()
        self.demucs = DemucsSettings()
        
        # Then load environment variables
        self._load_environment_variables()
        
        # Finally validate
        self._validate_settings()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # Audio settings
        self.audio.sample_rate = int(os.getenv('SYNTH_SAMPLE_RATE', 44100))
        self.audio.segment_length_ms = int(os.getenv('SYNTH_SEGMENT_LENGTH_MS', 5000))
        self.audio.stride_ms = int(os.getenv('SYNTH_STRIDE_MS', 2000))
        self.audio.random_ratio = float(os.getenv('SYNTH_RANDOM_RATIO', 0.3))
        
        # Spectral settings
        self.spectral.window_size = int(os.getenv('SYNTH_WINDOW_SIZE', 2048))
        self.spectral.hop_length = int(os.getenv('SYNTH_HOP_LENGTH', 512))
        self.spectral.n_fft = int(os.getenv('SYNTH_N_FFT', 2048))
        
        # Processing settings
        self.processing.batch_size = int(os.getenv('SYNTH_BATCH_SIZE', 4))
        self.processing.num_workers = int(os.getenv('SYNTH_NUM_WORKERS', 2))
        self.processing.use_gpu = os.getenv('SYNTH_USE_GPU', 'false').lower() == 'true'
        self.processing.max_memory_gb = float(os.getenv('SYNTH_MAX_MEMORY_GB', 8.0))
        
        # Demucs settings
        self.demucs.model_name = os.getenv('SYNTH_DEMUCS_MODEL', 'htdemucs')
        self.demucs.shifts = int(os.getenv('SYNTH_DEMUCS_SHIFTS', 1))
        self.demucs.split = os.getenv('SYNTH_DEMUCS_SPLIT', 'true').lower() == 'true'
        self.demucs.overlap = float(os.getenv('SYNTH_DEMUCS_OVERLAP', 0.25))
    
    def _validate_settings(self):
        """Validate that all settings are within acceptable ranges."""
        # Audio validation
        if self.audio.sample_rate not in [22050, 44100, 48000]:
            raise ValueError(f"Sample rate {self.audio.sample_rate} not supported")
        
        if self.audio.segment_length_ms < 1000 or self.audio.segment_length_ms > 30000:
            raise ValueError(f"Segment length {self.audio.segment_length_ms}ms out of range (1000-30000)")
        
        if self.audio.random_ratio < 0.0 or self.audio.random_ratio > 1.0:
            raise ValueError(f"Random ratio {self.audio.random_ratio} out of range (0.0-1.0)")
        
        # Spectral validation
        if self.spectral.window_size not in [512, 1024, 2048, 4096]:
            raise ValueError(f"Window size {self.spectral.window_size} not supported")
        
        if self.spectral.hop_length >= self.spectral.window_size:
            raise ValueError("Hop length must be less than window size")
        
        # Processing validation
        if self.processing.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        if self.processing.max_memory_gb < 1.0:
            raise ValueError("Max memory must be at least 1GB")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for logging/debugging."""
        return {
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'segment_length_ms': self.audio.segment_length_ms,
                'stride_ms': self.audio.stride_ms,
                'random_ratio': self.audio.random_ratio,
                'random_length_range': self.audio.random_length_range
            },
            'spectral': {
                'window_size': self.spectral.window_size,
                'hop_length': self.spectral.hop_length,
                'n_fft': self.spectral.n_fft,
                'frequency_bins': self.spectral.frequency_bins
            },
            'processing': {
                'batch_size': self.processing.batch_size,
                'num_workers': self.processing.num_workers,
                'use_gpu': self.processing.use_gpu,
                'max_memory_gb': self.processing.max_memory_gb
            },
            'demucs': {
                'model_name': self.demucs.model_name,
                'shifts': self.demucs.shifts,
                'split': self.demucs.split,
                'overlap': self.demucs.overlap
            }
        }
    
    def get_environment_template(self) -> str:
        """Generate a template .env file with current settings."""
        template = """# Audio Processing Pipeline Environment Configuration

# Project Paths
# SYNTH_PROJ_ROOT=/path/to/your/project  # Optional: Override project root
# SYNTH_EXTERNAL_DRIVE=/Volumes/YourDrive  # Optional: External drive path

# Data Directory Names
# SYNTH_SEGMENTED_AUDIO_DIR=segmented_audio
# SYNTH_SEGMENTED_STEMS_DIR=segmented_stems
# SYNTH_SPECTRAL_DATA_DIR=spectral_data
# SYNTH_RAW_AUDIO_DIR=BassVocalDrumGuitar

# Audio Settings
SYNTH_SAMPLE_RATE=44100
SYNTH_SEGMENT_LENGTH_MS=5000
SYNTH_STRIDE_MS=2000
SYNTH_RANDOM_RATIO=0.3

# Spectral Analysis Settings
SYNTH_WINDOW_SIZE=2048
SYNTH_HOP_LENGTH=512
SYNTH_N_FFT=2048

# Processing Settings
SYNTH_BATCH_SIZE=4
SYNTH_NUM_WORKERS=2
SYNTH_USE_GPU=false
SYNTH_MAX_MEMORY_GB=8.0

# Demucs Settings
SYNTH_DEMUCS_MODEL=htdemucs
SYNTH_DEMUCS_SHIFTS=1
SYNTH_DEMUCS_SPLIT=true
SYNTH_DEMUCS_OVERLAP=0.25

# Logging
SYNTH_LOG_LEVEL=INFO
"""
        return template

# Global settings instance
settings = Settings() 