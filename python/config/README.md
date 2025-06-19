# Data Path Configuration

This directory contains the configuration for data paths in the project. The system is designed to automatically use external drive data when available, with fallback to local paths.

## Overview

The project now uses a centralized path configuration system that automatically detects whether the external drive is mounted and routes data access accordingly.

## Files

- `paths.py`: Main configuration file containing all data path definitions and helper functions

## Data Locations

### External Drive (Primary)
When the external drive is mounted, data is accessed from:
- `/Volumes/Extreme SSD/segmented_audio_gen1`
- `/Volumes/Extreme SSD/segmented_stems_gen1`
- `/Volumes/Extreme SSD/spectral_data_gen1`

### Local Fallback
If the external drive is not available, data falls back to:
- `../segmented_audio` (symlinked to external drive when available)
- `../segmented_stems` (symlinked to external drive when available)
- `../spectral_data` (symlinked to external drive when available)

## Usage

### Basic Usage
```python
from paths import get_segmented_audio_path, get_segmented_stems_path, get_spectral_data_path

# Get paths
audio_path = get_segmented_audio_path()
stems_path = get_segmented_stems_path()
spectral_path = get_spectral_data_path()
```

### Advanced Usage
```python
from paths import get_data_paths, check_external_drive_available

# Check if external drive is available
if check_external_drive_available():
    print("Using external drive data")
else:
    print("Using local data")

# Get all paths at once
paths = get_data_paths(use_external=True)
print(paths['segmented_audio'])
print(paths['segmented_stems'])
print(paths['spectral_data'])
```

## Symlinks

The project directory contains symlinks that point to the external drive data:
- `segmented_audio` → `/Volumes/Extreme SSD/segmented_audio_gen1`
- `segmented_stems` → `/Volumes/Extreme SSD/segmented_stems_gen1`
- `spectral_data` → `/Volumes/Extreme SSD/spectral_data_gen1`

These symlinks ensure that existing code continues to work without modification while accessing the external drive data.

## Benefits

1. **Automatic Detection**: The system automatically detects whether the external drive is mounted
2. **Fallback Support**: If the external drive is not available, it falls back to local paths
3. **Centralized Configuration**: All paths are managed in one place
4. **Backward Compatibility**: Existing code continues to work through symlinks
5. **Clean Project Directory**: The project directory is clean and doesn't contain large data files

## Notes

- The external drive must be mounted at `/Volumes/Extreme SSD` for automatic detection
- Data on the external drive uses the `_gen1` suffix to distinguish it from other versions
- The system will automatically create directories if they don't exist
- All Python scripts have been updated to use this configuration system 