# Audio Processing Pipeline - Python

A comprehensive audio processing pipeline that transforms raw music files into detailed spectral analysis data for neural network training and visualization.

## 🎯 Overview

This pipeline processes audio through three main stages:

1. **Audio Segmentation** - Breaks full tracks into manageable segments
2. **Stem Separation** - Isolates individual musical components (vocals, drums, bass, other)
3. **Spectral Analysis** - Converts audio into frequency-domain "Bytes" using STFT

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kadennichols5/synthProj.git

# Navigate to the project directory
cd synthProj

# Install Python dependencies
cd python
pip install -r requirements.txt

# Install Demucs for audio separation
pip install demucs

# Run the setup script to verify everything works
python setup.py
```

### Basic Usage

#### Run the Complete Pipeline
```bash
# Process all audio files in the default input directory
python src/main.py --mode full
```

#### Run Individual Stages
```bash
# Only segment audio files
python src/main.py --mode segment

# Only separate stems (requires segmented audio)
python src/main.py --mode separate

# Only spectral analysis (requires separated stems)
python src/main.py --mode spectral
```

#### Demo with a Single File
```bash
# Run demo on a single audio file
python examples/basic_pipeline_demo.py /path/to/your/song.mp3
```

## 📁 Project Structure

```
synthProj/                        # Project root
├── BassVocalDrumGuitar/          # Raw audio files (add your music here)
├── python/                       # Python pipeline code
│   ├── src/                      # Core source code
│   │   ├── main.py               # Main pipeline entry point
│   │   ├── segment_generator.py  # Audio segmentation
│   │   ├── audio_separation.py   # Stem separation using Demucs
│   │   ├── spectral_analysis.py  # STFT and Byte generation
│   │   ├── process_stems.py      # Batch spectral processing
│   │   ├── neural_network.py     # Neural network models
│   │   └── batch_separate_segments.py # Batch processing utilities
│   ├── config/                   # Configuration files
│   │   └── paths.py              # Data path management
│   ├── examples/                 # Example scripts
│   │   └── basic_pipeline_demo.py # Basic pipeline demonstration
│   ├── tests/                    # Unit tests
│   ├── requirements.txt          # Python dependencies
│   ├── setup.py                  # Setup and verification script
│   └── README.md                 # This file
├── segmented_audio/              # Generated audio segments
├── segmented_stems/              # Generated separated stems
└── spectral_data/                # Generated spectral analysis results
```

## 🔧 Configuration

### Data Paths

The project uses a centralized path configuration system in `config/paths.py`:

- **External Drive Support**: Automatically detects external drive at `/Volumes/Extreme SSD`
- **Fallback Paths**: Uses local paths if external drive unavailable
- **Flexible Storage**: Supports both local and external storage seamlessly

### Key Paths

```python
from config.paths import (
    get_segmented_audio_path,
    get_segmented_stems_path, 
    get_spectral_data_path
)

# Get configured paths
segmented_audio = get_segmented_audio_path()
segmented_stems = get_segmented_stems_path()
spectral_data = get_spectral_data_path()
```

## 🎵 Pipeline Components

### 1. Audio Segmentation (`segment_generator.py`)

Breaks full music tracks into smaller segments for processing.

**Features**:
- **Symmetric Sampling**: 70% regular stride-based segments
- **Random Sampling**: 30% random start positions and lengths
- **Configurable Parameters**: Segment length, overlap, random ratio
- **Metadata Tracking**: Saves segment information and timing

**Usage**:
```python
from segment_generator import extract_segments

extract_segments(
    "song.mp3",
    segment_len_ms=5000,    # 5 second segments
    stride_ms=2000,         # 2 second overlap
    random_ratio=0.3,       # 30% random segments
    output_dir="segments"
)
```

### 2. Stem Separation (`audio_separation.py`)

Separates audio segments into individual musical components using Demucs.

**Features**:
- **Multiple Stems**: vocals, drums, bass, other
- **High Quality**: Uses htdemucs model
- **Batch Processing**: Efficient multi-file processing
- **Error Handling**: Robust error recovery

**Usage**:
```python
from audio_separation import AudioSeparator

separator = AudioSeparator("segment.wav", "output_dir")
separator.separate_audio()
```

### 3. Spectral Analysis (`spectral_analysis.py`)

Converts audio stems into frequency-domain "Bytes" using STFT.

**Features**:
- **STFT Processing**: Configurable window size and hop length
- **Byte Objects**: Structured spectral data
- **Normalization**: Automatic normalization for neural networks
- **Frequency Analysis**: Energy analysis across frequency bands

**Usage**:
```python
from spectral_analysis import ByteExtractor

extractor = ByteExtractor()
bytes_list = extractor.extract_bytes(audio_data, "vocals")

# Access spectral information
for byte in bytes_list:
    print(f"Time: {byte.time}")
    print(f"Magnitudes: {len(byte.magnitudes)} frequency bins")
    print(f"Phases: {len(byte.phases)} frequency bins")
```

## 📊 Data Formats

### Spectral Byte Structure

Each "Byte" contains complete spectral information for a time window:

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

### Output JSON Format

Spectral data is saved in structured JSON format:

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

## 🧠 Neural Network Integration

The spectral data feeds into neural networks for visualization:

### VisualEncoder

Takes spectral bytes and outputs visual parameters:

```python
from neural_network import VisualEncoder

encoder = VisualEncoder(input_dim=4096)  # 2048 magnitudes + 2048 phases
visual_params = encoder(spectral_vector)

# Output parameters:
# - shape: [6] - Shape categories
# - motion: [6] - Velocity and acceleration  
# - texture: [8] - Texture parameters
# - color: [4] - RGBA values
# - brightness: [1] - Brightness scalar
# - position: [3] - XYZ coordinates
# - pattern: [6] - Pattern categories
```

## 🚀 Advanced Usage

### Programmatic Pipeline

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

### Custom Parameters

```python
# Custom segmentation parameters
extract_segments(
    "song.mp3",
    segment_len_ms=3000,    # 3 second segments
    stride_ms=1000,         # 1 second overlap
    random_ratio=0.5,       # 50% random segments
    random_len_range=(2000, 5000)
)

# Custom spectral analysis parameters
extractor = ByteExtractor(
    window_size=1024,       # Smaller window for higher time resolution
    hop_length=256,         # Smaller hop for more overlap
    sample_rate=44100
)
```

## 📈 Performance

### Processing Times (approximate)
- **Segmentation**: ~1-2 minutes per song
- **Separation**: ~5-10 minutes per segment
- **Spectral Analysis**: ~1-2 minutes per segment

### Storage Requirements
- **Segmented Audio**: ~50-100MB per song
- **Separated Stems**: ~200-400MB per song  
- **Spectral Data**: ~50-100MB per song (JSON)

### Memory Usage
- Processes files sequentially to manage memory
- Configurable batch sizes for optimization
- External drive support for large datasets

## 🐛 Troubleshooting

### Common Issues

1. **Demucs not found**
   ```bash
   pip install demucs
   ```

2. **Memory errors**
   - Reduce segment length
   - Process fewer files at once
   - Use smaller STFT window size

3. **External drive not mounted**
   - Check mount point in `config/paths.py`
   - Verify drive is accessible

4. **Audio format issues**
   - Ensure files are valid MP3/WAV
   - Check file permissions

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/main.py --mode demo --demo_file test.wav
```

### Log Files

- **Main log**: `audio_pipeline.log`
- **Separation errors**: `separation_errors.txt`
- **Processed files**: `separated_files.txt`

## 🧪 Testing

Run the test suite:

```bash
cd python
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- **Complete Pipeline Guide**: `PROJECT_PIPELINE.md`
- **API Documentation**: Generated from docstrings
- **Examples**: `examples/basic_pipeline_demo.py`

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Use meaningful commit messages

## 📄 License

MIT License - See main project README for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the log files
3. Run the demo script to verify setup
4. Check the detailed pipeline documentation

---

**Happy Audio Processing! 🎵** 