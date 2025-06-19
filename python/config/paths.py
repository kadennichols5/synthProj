"""
Configuration file for data paths.
This file centralizes all data directory paths to make them easily configurable.
"""

import os
from pathlib import Path

# Base project directory
PROJECT_ROOT = Path("/Users/kadensnichols/Desktop/synthProj")

# External drive mount point
EXTERNAL_DRIVE = Path("/Volumes/Extreme SSD")

# Data directories on external drive (with gen1 suffix)
EXTERNAL_SEGMENTED_AUDIO = EXTERNAL_DRIVE / "segmented_audio_gen1"
EXTERNAL_SEGMENTED_STEMS = EXTERNAL_DRIVE / "segmented_stems_gen1"
EXTERNAL_SPECTRAL_DATA = EXTERNAL_DRIVE / "spectral_data_gen1"

# Legacy local paths (for reference and fallback)
LOCAL_SEGMENTED_AUDIO = PROJECT_ROOT / "segmented_audio"
LOCAL_SEGMENTED_STEMS = PROJECT_ROOT / "segmented_stems"
LOCAL_SPECTRAL_DATA = PROJECT_ROOT / "spectral_data"

def get_data_paths(use_external=True):
    """
    Get the appropriate data paths based on whether external drive is available.
    
    Args:
        use_external (bool): Whether to use external drive paths if available
        
    Returns:
        dict: Dictionary containing the data paths
    """
    if use_external and EXTERNAL_DRIVE.exists():
        return {
            'segmented_audio': str(EXTERNAL_SEGMENTED_AUDIO),
            'segmented_stems': str(EXTERNAL_SEGMENTED_STEMS),
            'spectral_data': str(EXTERNAL_SPECTRAL_DATA)
        }
    else:
        return {
            'segmented_audio': str(LOCAL_SEGMENTED_AUDIO),
            'segmented_stems': str(LOCAL_SEGMENTED_STEMS),
            'spectral_data': str(LOCAL_SPECTRAL_DATA)
        }

def check_external_drive_available():
    """Check if the external drive is mounted and accessible."""
    return EXTERNAL_DRIVE.exists()

def get_segmented_audio_path():
    """Get the path to segmented audio data."""
    paths = get_data_paths()
    return paths['segmented_audio']

def get_segmented_stems_path():
    """Get the path to segmented stems data."""
    paths = get_data_paths()
    return paths['segmented_stems']

def get_spectral_data_path():
    """Get the path to spectral data."""
    paths = get_data_paths()
    return paths['spectral_data'] 