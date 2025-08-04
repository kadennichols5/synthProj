"""
Configuration file for data paths.
This file centralizes all data directory paths to make them easily configurable.
Supports both local and external storage with automatic detection.
"""

import os
from pathlib import Path
from typing import Dict, Optional

# Get project root dynamically (relative to this config file)
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent.parent  # Go up from config/python/project_root

# Environment variable for custom project root
ENV_PROJECT_ROOT = os.getenv('SYNTH_PROJ_ROOT')
if ENV_PROJECT_ROOT:
    PROJECT_ROOT = Path(ENV_PROJECT_ROOT)

# External drive configuration (configurable via environment)
EXTERNAL_DRIVE_PATH = os.getenv('SYNTH_EXTERNAL_DRIVE', '')
EXTERNAL_DRIVE = Path(EXTERNAL_DRIVE_PATH) if EXTERNAL_DRIVE_PATH else None

# Data directory names (configurable via environment)
SEGMENTED_AUDIO_DIR = os.getenv('SYNTH_SEGMENTED_AUDIO_DIR', 'segmented_audio')
SEGMENTED_STEMS_DIR = os.getenv('SYNTH_SEGMENTED_STEMS_DIR', 'segmented_stems')
SPECTRAL_DATA_DIR = os.getenv('SYNTH_SPECTRAL_DATA_DIR', 'spectral_data')
NEURAL_OUTPUT_DIR = os.getenv('SYNTH_NEURAL_OUTPUT_DIR', 'neural_output')
RAW_AUDIO_DIR = os.getenv('SYNTH_RAW_AUDIO_DIR', 'BassVocalDrumGuitar')

# Generation suffix for versioning (configurable via environment)
DATA_GENERATION_SUFFIX = os.getenv('SYNTH_DATA_GENERATION_SUFFIX', '_gen1')

# External drive paths (with generation suffix for versioning)
if EXTERNAL_DRIVE:
    EXTERNAL_SEGMENTED_AUDIO = EXTERNAL_DRIVE / f"{SEGMENTED_AUDIO_DIR}{DATA_GENERATION_SUFFIX}"
    EXTERNAL_SEGMENTED_STEMS = EXTERNAL_DRIVE / f"{SEGMENTED_STEMS_DIR}{DATA_GENERATION_SUFFIX}"
    EXTERNAL_SPECTRAL_DATA = EXTERNAL_DRIVE / f"{SPECTRAL_DATA_DIR}{DATA_GENERATION_SUFFIX}"
    EXTERNAL_NEURAL_OUTPUT = EXTERNAL_DRIVE / f"{NEURAL_OUTPUT_DIR}{DATA_GENERATION_SUFFIX}"
else:
    EXTERNAL_SEGMENTED_AUDIO = None
    EXTERNAL_SEGMENTED_STEMS = None
    EXTERNAL_SPECTRAL_DATA = None
    EXTERNAL_NEURAL_OUTPUT = None

# Local paths (relative to project root)
LOCAL_SEGMENTED_AUDIO = PROJECT_ROOT / SEGMENTED_AUDIO_DIR
LOCAL_SEGMENTED_STEMS = PROJECT_ROOT / SEGMENTED_STEMS_DIR
LOCAL_SPECTRAL_DATA = PROJECT_ROOT / SPECTRAL_DATA_DIR
LOCAL_NEURAL_OUTPUT = PROJECT_ROOT / NEURAL_OUTPUT_DIR
LOCAL_RAW_AUDIO = PROJECT_ROOT / RAW_AUDIO_DIR

def get_data_paths(use_external: bool = True) -> Dict[str, str]:
    """
    Get the appropriate data paths based on whether external drive is available.
    
    Args:
        use_external (bool): Whether to use external drive paths if available
        
    Returns:
        dict: Dictionary containing the data paths
    """
    if use_external and EXTERNAL_DRIVE and EXTERNAL_DRIVE.exists():
        return {
            'segmented_audio': str(EXTERNAL_SEGMENTED_AUDIO),
            'segmented_stems': str(EXTERNAL_SEGMENTED_STEMS),
            'spectral_data': str(EXTERNAL_SPECTRAL_DATA),
            'neural_output': str(EXTERNAL_NEURAL_OUTPUT),
            'raw_audio': str(LOCAL_RAW_AUDIO)  # Raw audio always local
        }
    else:
        return {
            'segmented_audio': str(LOCAL_SEGMENTED_AUDIO),
            'segmented_stems': str(LOCAL_SEGMENTED_STEMS),
            'spectral_data': str(LOCAL_SPECTRAL_DATA),
            'neural_output': str(LOCAL_NEURAL_OUTPUT),
            'raw_audio': str(LOCAL_RAW_AUDIO)
        }

def check_external_drive_available() -> bool:
    """Check if the external drive is mounted and accessible."""
    return EXTERNAL_DRIVE is not None and EXTERNAL_DRIVE.exists()

def get_segmented_audio_path() -> str:
    """Get the path to segmented audio data."""
    paths = get_data_paths()
    return paths['segmented_audio']

def get_segmented_stems_path() -> str:
    """Get the path to segmented stems data."""
    paths = get_data_paths()
    return paths['segmented_stems']

def get_spectral_data_path() -> str:
    """Get the path to spectral data."""
    paths = get_data_paths()
    return paths['spectral_data']

def get_neural_output_path() -> str:
    """Get the path to neural output data."""
    paths = get_data_paths()
    return paths['neural_output']

def get_raw_audio_path() -> str:
    """Get the path to raw audio data."""
    paths = get_data_paths()
    return paths['raw_audio']

def ensure_directories_exist() -> None:
    """Create all necessary directories if they don't exist."""
    paths = get_data_paths()
    for path_name, path_str in paths.items():
        Path(path_str).mkdir(parents=True, exist_ok=True)

def get_project_info() -> Dict[str, str]:
    """Get project information and current configuration."""
    return {
        'project_root': str(PROJECT_ROOT),
        'external_drive_available': str(check_external_drive_available()),
        'external_drive_path': str(EXTERNAL_DRIVE) if EXTERNAL_DRIVE else 'N/A',
        'using_external_storage': str(check_external_drive_available()),
        'segmented_audio_path': get_segmented_audio_path(),
        'segmented_stems_path': get_segmented_stems_path(),
        'spectral_data_path': get_spectral_data_path(),
        'neural_output_path': get_neural_output_path(),
        'raw_audio_path': get_raw_audio_path()
    } 