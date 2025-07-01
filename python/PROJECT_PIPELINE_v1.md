# Audio Processing Pipeline - Complete Documentation

## Overview

This project implements a comprehensive audio processing pipeline that transforms raw music files into detailed spectral analysis data. The pipeline consists of three main stages:

1. **Audio Segmentation** - Breaking full tracks into smaller, manageable segments
2. **Stem Separation** - Isolating individual musical components (vocals, drums, bass, other)
3. **Spectral Analysis** - Converting audio into frequency-domain "Bytes" using Short-Time Fourier Transforms

## Pipeline Architecture

```
Raw Audio Files (.mp3/.wav)
           ↓
    [Audio Segmentation]
           ↓
    Segmented Audio (.wav)
           ↓
    [Stem Separation]
           ↓
    Separated Stems (vocals, drums, bass, other)
           ↓
    [Spectral Analysis]
           ↓
    Spectral Bytes (JSON format)
```

## Core Components

### 1. Audio Segmentation (`segment_generator.py`)

**Purpose**: Breaks full music tracks into smaller segments for processing.

**Key Features**:
- **Symmetric Sampling**: 70% of segments use regular stride-based sampling
- **Random Sampling**: 30% of segments use random start positions and lengths
- **Overlap Control**: Configurable stride and segment lengths
- **Metadata Tracking**: Saves segment information including timing and type

**Parameters**:
- `segment_len_ms`: Length of each segment (default: 5000ms)
- `stride_ms`: Overlap between consecutive segments (default: 2000ms)
- `random_ratio`: Proportion of random segments (default: 0.3)
- `random_len_range`: Range for random segment lengths (default: 3000-7000ms)

**Output**: WAV files with metadata JSON

### 2. Stem Separation (`audio_separation.py`)

**Purpose**: Separates audio segments into individual musical components using Demucs.

**Key Features**:
- **Multiple Stems**: Separates into vocals, drums, bass, and other
- **High Quality**: Uses htdemucs model for best separation quality
- **Batch Processing**: Can process multiple files efficiently
- **Error Handling**: Robust error handling and logging

**Stems Generated**:
- `vocals.mp3` - Vocal tracks
- `drums.mp3` - Drum tracks  
- `bass.mp3` - Bass tracks
- `other.mp3` - All other instruments

**Output**: Organized directory structure with separated stems

### 3. Spectral Analysis (`spectral_analysis.py`)

**Purpose**: Converts audio stems into frequency-domain "Bytes" using STFT.

**Key Features**:
- **STFT Processing**: Short-Time Fourier Transform with configurable parameters
- **Byte Objects**: Structured data containing magnitude, phase, and frequency information
- **Normalization**: Automatic normalization for neural network input
- **Frequency Bands**: Energy analysis across different frequency ranges

**Byte Structure**:
```python
@dataclass
class Byte:
    time: float                    # Time index
    stem_label: str               # Stem type (vocals, drums, etc.)
    magnitudes: np.ndarray        # Frequency magnitudes
    phases: np.ndarray           # Frequency phases
    frequency_bins: np.ndarray   # Frequency bin centers
    sample_rate: int             # Audio sample rate
```

**STFT Parameters**:
- `window_size`: 2048 samples (frequency resolution)
- `hop_length`: 512 samples (time resolution)
- `sample_rate`: 44100 Hz

### 4. Data Processing (`process_stems.py`)

**Purpose**: Orchestrates the spectral analysis of all separated stems.

**Key Features**:
- **Batch Processing**: Processes all stems in a directory
- **JSON Output**: Saves spectral data in structured JSON format
- **Error Recovery**: Continues processing even if individual files fail
- **Memory Efficient**: Processes files one at a time

**Output Format**:
```json
{
    "segment_name": "song_segment_0001",
    "stems": {
        "vocals": [
            {
                "time": 0.0,
                "stem": "vocals",
                "magnitudes": [...],
                "phases": [...],
                "frequencies": [...],
                "sample_rate": 44100
            }
        ],
        "drums": [...],
        "bass": [...],
        "other": [...]
    }
}
```

## Usage Examples

### Running the Complete Pipeline

```bash
# Run the full pipeline
python src/main.py --mode full

# Run individual stages
python src/main.py --mode segment
python src/main.py --mode separate  
python src/main.py --mode spectral

# Demo spectral analysis on a single file
python src/main.py --mode demo --demo_file path/to/audio.wav
```

### Programmatic Usage

```python
from src.main import AudioProcessingPipeline

# Initialize pipeline
pipeline = AudioProcessingPipeline(
    input_dir="/path/to/raw/audio",
    output_dir="/path/to/output"
)

# Run complete pipeline
pipeline.run_full_pipeline()

# Or run individual stages
pipeline.segment_audio_files()
pipeline.separate_stems()
pipeline.generate_spectral_data()
```

### Direct Module Usage

```python
# Segment audio
from segment_generator import extract_segments
extract_segments("song.mp3", output_dir="segments")

# Separate stems
from audio_separation import AudioSeparator
separator = AudioSeparator("segment.wav", "output_dir")
separator.separate_audio()

# Spectral analysis
from spectral_analysis import ByteExtractor
extractor = ByteExtractor()
bytes_list = extractor.extract_bytes(audio_data, "vocals")
```

## Data Flow and Storage

### Directory Structure
```
project_root/
├── BassVocalDrumGuitar/          # Raw audio files
├── segmented_audio/              # Segmented audio files
├── segmented_stems/              # Separated stems
├── spectral_data/               # Spectral analysis results
└── python/
    ├── src/                     # Source code
    ├── config/                  # Configuration
    └── tests/                   # Unit tests
```

### External Drive Integration
The project supports external drive storage for large datasets:
- Automatically detects external drive at `/Volumes/Extreme SSD`
- Uses external paths when available: `segmented_audio_gen1/`, `segmented_stems_gen1/`, `spectral_data_gen1/`
- Falls back to local paths if external drive unavailable

## Performance Considerations

### Memory Usage
- **Segmentation**: Processes one file at a time
- **Separation**: Uses Demucs with memory-efficient settings
- **Spectral Analysis**: Processes stems sequentially to manage memory

### Processing Time
- **Segmentation**: ~1-2 minutes per song
- **Separation**: ~5-10 minutes per segment (depends on length)
- **Spectral Analysis**: ~1-2 minutes per segment

### Storage Requirements
- **Segmented Audio**: ~50-100MB per song
- **Separated Stems**: ~200-400MB per song
- **Spectral Data**: ~50-100MB per song (JSON format)

## Neural Network Integration

The spectral data is designed to feed into neural networks for visualization:

### VisualEncoder (`neural_network.py`)
- Takes spectral bytes as input
- Outputs visual parameters: shape, motion, texture, color, brightness, position, pattern
- Supports both supervised and unsupervised training

### Input Format
```python
# Normalized spectral vector
byte_vector = byte.get_vector()  # Concatenated magnitudes + phases
```

### Output Format
```python
{
    'shape': [6],      # Shape categories
    'motion': [6],     # Velocity and acceleration
    'texture': [8],    # Texture parameters
    'color': [4],      # RGBA values
    'brightness': [1], # Brightness scalar
    'position': [3],   # XYZ coordinates
    'pattern': [6]     # Pattern categories
}
```

## Error Handling and Logging

### Logging
- Comprehensive logging to both file and console
- Different log levels for debugging and production
- Log file: `audio_pipeline.log`

### Error Recovery
- Continues processing even if individual files fail
- Tracks completed files to avoid reprocessing
- Error logs: `separation_errors.txt`

### Validation
- Input file validation
- Output directory creation
- Data integrity checks

## Future Enhancements

### Planned Features
- **Real-time Processing**: Stream processing for live audio
- **GPU Acceleration**: CUDA support for faster processing
- **Advanced Visualization**: Real-time visual feedback
- **Model Training**: Automated model training pipeline
- **API Interface**: REST API for remote processing

### Optimization Opportunities
- **Parallel Processing**: Multi-threaded stem separation
- **Compression**: Efficient data storage formats
- **Caching**: Intermediate result caching
- **Incremental Processing**: Process only new/changed files

## Troubleshooting

### Common Issues
1. **Demucs not found**: Install with `pip install demucs`
2. **Memory errors**: Reduce batch size or use smaller segments
3. **External drive not mounted**: Check mount point in `paths.py`
4. **Audio format issues**: Ensure files are valid MP3/WAV

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/main.py --mode demo --demo_file test.wav
```

This pipeline provides a robust foundation for audio-to-visual processing, combining state-of-the-art audio separation with detailed spectral analysis to create rich, structured data for neural network training and real-time visualization. 