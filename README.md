# Audio Stem Processing and Spectral Analysis Project

This project processes audio stems and performs spectral analysis on music tracks, generating detailed spectral data for various musical components.

## Project Structure

```
.
├── python/
│   ├── config/
│   │   ├── paths.py                  # Data path configuration
│   │   └── README.md                 # Configuration documentation
│   └── src/
│       ├── process_stems.py          # Main stem processing script
│       ├── spectral_analysis.py      # Spectral analysis utilities
│       ├── audio_separation.py       # Audio stem separation functionality
│       ├── batch_separate_segments.py # Batch processing for audio segments
│       ├── segment_generator.py      # Generates audio segments from full tracks
│       ├── gen1_data_cleaning.py     # Data cleaning utilities
│       ├── neural_network.py         # Neural network implementation
│       ├── main.py                   # Main entry point
│       └── __init__.py              # Package initialization
├── segmented_audio/                  # Symlink to external drive data
├── segmented_stems/                  # Symlink to external drive data
└── spectral_data/                   # Symlink to external drive data
```

## Data Management

### External Drive Setup
The project is configured to use an external drive for data storage to manage large file sizes efficiently:

- **External Drive**: `/Volumes/Extreme SSD`
- **Data Locations**: 
  - `segmented_audio_gen1/` - Audio segments
  - `segmented_stems_gen1/` - Separated audio stems
  - `spectral_data_gen1/` - Spectral analysis results

### Automatic Path Configuration
The project uses a centralized path configuration system that:
- Automatically detects when the external drive is mounted
- Routes data access to external drive when available
- Falls back to local paths if external drive is unavailable
- Maintains backward compatibility through symlinks

### Usage
```python
from python.config.paths import get_segmented_audio_path, get_segmented_stems_path, get_spectral_data_path

# Get configured paths
audio_path = get_segmented_audio_path()
stems_path = get_segmented_stems_path()
spectral_path = get_spectral_data_path()
```

## Features

- Processes audio stems (vocals, drums, bass, other) from music tracks
- Supports both MP3 and WAV file formats
- Performs spectral analysis on audio segments
- Generates detailed spectral data including:
  - Magnitudes
  - Phases
  - Frequency bins
  - Time information
  - Sample rates
- Automatic external drive detection and data routing
- Centralized path configuration

## Requirements

- Python 3.x
- External drive mounted at `/Volumes/Extreme SSD` (for optimal performance)
- Required Python packages:
  - numpy
  - soundfile
  - pydub
  - (Additional dependencies listed in requirements.txt)

## Output Format

The spectral data is saved in JSON format with the following structure:
```json
{
    "segment_name": "song_name",
    "stems": {
        "vocals": [...],
        "drums": [...],
        "bass": [...],
        "other": [...]
    }
}
```

Each stem contains detailed spectral information including:
- Time markers
- Magnitude values
- Phase values
- Frequency bins
- Sample rate information

## Data Processing Pipeline

1. **Audio Segmentation**: Full tracks are segmented into smaller chunks
2. **Stem Separation**: Each segment is separated into individual stems (vocals, drums, bass, other)
3. **Spectral Analysis**: Detailed spectral analysis is performed on each stem
4. **Data Storage**: Results are stored on external drive with `_gen1` suffix

## Notes

- The project currently processes a variety of music tracks including:
  - Classic rock songs
  - Alternative rock
  - Indie tracks
- Each processed file generates spectral data files ranging from ~96MB to ~182MB, ~300GB in total
- The spectral analysis preserves important audio characteristics while providing detailed frequency domain information
- Data is automatically stored on external drive when available to manage storage efficiently
- Symlinks in the project directory provide seamless access to external data

## License

MIT License - Copyright (c) 2025 Kaden Nichols



