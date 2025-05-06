# Audio Stem Processing and Spectral Analysis Project

This project processes audio stems and performs spectral analysis on music tracks, generating detailed spectral data for various musical components.

## Project Structure

```
.
├── python/
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
├── segmented_stems/                 # Directory containing processed audio stems
└── spectral_data/                  # Directory containing spectral analysis results
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

## Requirements

- Python 3.x
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

## Notes

- The project currently processes a variety of music tracks including:
  - Classic rock songs
  - Alternative rock
  - Indie tracks
- Each processed file generates spectral data files ranging from ~96MB to ~182MB, ~300GB in total.
- The spectral analysis preserves important audio characteristics while providing detailed frequency domain information

## License

MIT License - Copyright (c) 2025 Kaden Nichols



