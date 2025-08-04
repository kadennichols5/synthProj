# Audio Processing Pipeline - Python

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive audio processing pipeline that transforms raw music files into detailed spectral analysis data for neural network training and real-time visualization.

## 🎯 Overview

This pipeline processes audio through three main stages:

1. **Audio Segmentation** - Breaks full tracks into manageable segments using dual sampling strategies
2. **Stem Separation** - Isolates individual musical components (vocals, drums, bass, other)
3. **Spectral Analysis** - Converts audio into frequency-domain data using STFT

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthProj.git

# Navigate to the project directory
cd synthProj/python

# Install Python dependencies
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

# Or specify a custom input directory
python src/main.py --mode full --input /path/to/your/audio/files
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
python/                       # Python pipeline code
├── src/                      # Core source code
│   ├── main.py               # Main pipeline entry point
│   ├── segment_generator.py  # Audio segmentation with dual sampling
│   ├── audio_separation.py   # Stem separation using Demucs
│   ├── spectral_analysis.py  # STFT and spectral data generation
│   ├── process_stems.py      # Batch spectral processing
│   ├── neural_network.py     # Neural network models
│   ├── batch_separate_segments.py # Batch processing utilities
│   ├── train_visual_encoder.py # Visual encoder training
│   ├── generate_unity_data.py # Unity data generation
│   ├── unity_data_generator.py # Unity-specific data processing
│   └── test_model_outputs.py # Model testing utilities
├── config/                   # Configuration files
│   └── paths.py              # Data path management
├── examples/                 # Example scripts
│   └── basic_pipeline_demo.py # Basic pipeline demonstration
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup and verification script
└── README.md                 # This file
```

## 🔧 Configuration

### Data Paths

The project uses a centralized path configuration system in `config/paths.py`:

- **Flexible Storage**: Supports both local and external storage
- **Environment Variables**: Configurable via environment variables
- **Automatic Detection**: Detects available storage options
- **Fallback System**: Graceful degradation when external storage unavailable

### Environment Variables

```bash
# External drive configuration (optional)
export SYNTH_EXTERNAL_DRIVE="/path/to/external/drive"

# Data directory customization
export SYNTH_SEGMENTED_AUDIO_DIR="segmented_audio"
export SYNTH_SEGMENTED_STEMS_DIR="segmented_stems"
export SYNTH_SPECTRAL_DATA_DIR="spectral_data"

# Generation versioning
export SYNTH_DATA_GENERATION_SUFFIX="_gen1"
```

### Usage
```python
from config.paths import get_segmented_audio_path, get_segmented_stems_path, get_spectral_data_path

# Get configured paths
audio_path = get_segmented_audio_path()
stems_path = get_segmented_stems_path()
spectral_path = get_spectral_data_path()
```

## 🎵 Audio Processing Features

### Dual Sampling Strategy
The segmentation system uses two sampling approaches:
- **70% Symmetric**: Regular stride-based sampling (5-second segments, 2-second overlap)
- **30% Random**: Random start positions and variable lengths (3-7 second segments)

### Stem Separation
- **High Quality**: Uses htdemucs model for best separation quality
- **Multiple Stems**: Separates into vocals, drums, bass, and other instruments
- **Batch Processing**: Efficient processing of multiple files
- **Error Handling**: Robust error handling and logging

### Spectral Analysis
- **STFT Processing**: Short-Time Fourier Transform with configurable parameters
- **Structured Data**: Frequency-domain information in organized Byte objects
- **Normalization**: Automatic normalization for neural network input
- **Time Resolution**: High temporal resolution for real-time visualization

## 📊 Output Format

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

## 🔄 Data Processing Pipeline

1. **Audio Segmentation**: Full tracks are segmented using dual sampling strategy
2. **Stem Separation**: Each segment is separated into individual stems (vocals, drums, bass, other)
3. **Spectral Analysis**: Detailed spectral analysis is performed on each stem
4. **Data Storage**: Results are stored in configurable locations with versioning support

## 🧠 Neural Network Integration

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

## 🛠️ Advanced Usage

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

## 🐛 Troubleshooting

### Common Issues
1. **Demucs not found**: Install with `pip install demucs`
2. **Memory errors**: Reduce batch size or use smaller segments
3. **Audio format issues**: Ensure files are valid MP3/WAV
4. **Path configuration**: Check environment variables and path settings

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/main.py --mode demo --demo_file test.wav
```

## 📈 Performance Considerations

### Processing Time Estimates
- **Segmentation**: ~1-2 minutes per song
- **Separation**: ~5-10 minutes per segment (depends on length)
- **Spectral Analysis**: ~1-2 minutes per segment

### Storage Requirements
- **Segmented Audio**: ~50-100MB per song
- **Separated Stems**: ~200-400MB per song
- **Spectral Data**: ~50-100MB per song (JSON format)

### Memory Optimization
- **Segmentation**: Processes one file at a time
- **Separation**: Uses Demucs with memory-efficient settings
- **Spectral Analysis**: Processes stems sequentially to manage memory

## 🎓 Learning Value

This pipeline demonstrates:
- **Audio Signal Processing**: STFT, spectral analysis, stem separation
- **Data Pipeline Design**: Modular, configurable processing systems
- **Performance Optimization**: Efficient audio processing techniques
- **Error Handling**: Robust error handling and logging
- **Configuration Management**: Flexible path and parameter configuration

## 📄 License

MIT License - See [LICENSE](../../LICENSE) file for details.

---

**Built with ❤️ for shapes and sounds and data science** 