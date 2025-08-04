# Synesthetic Audio Visualization System - Architecture

## Overview

This project implements a comprehensive audio processing pipeline that transforms raw music files into detailed spectral analysis data for real-time 3D visualization. The system demonstrates advanced audio processing, data transformation, and interactive visualization techniques.

## System Architecture

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
    Spectral Data (JSON format)
           ↓
    [Unity Visualization]
           ↓
    Real-time 3D Visuals
```

## Core Components

### 1. Audio Segmentation (`segment_generator.py`)

**Purpose**: Breaks full music tracks into smaller segments using dual sampling strategies.

**Key Features**:
- **Dual Sampling Strategy**: 
  - **70% Symmetric Sampling**: Regular stride-based sampling with configurable overlap
  - **30% Random Sampling**: Random start positions and variable lengths
- **Overlap Control**: Configurable stride and segment lengths
- **Metadata Tracking**: Saves segment information including timing and sampling type
- **Quality Assurance**: Avoids excessive overlap between symmetric and random segments

**Parameters**:
- `segment_len_ms`: Length of symmetric segments (default: 5000ms)
- `stride_ms`: Overlap between consecutive segments (default: 2000ms)
- `random_ratio`: Proportion of random segments (default: 0.3)
- `random_len_range`: Range for random segment lengths (default: 3000-7000ms)

**Output**: WAV files with comprehensive metadata JSON

### 2. Stem Separation (`audio_separation.py`)

**Purpose**: Separates audio segments into individual musical components using Demucs.

**Key Features**:
- **Multiple Stems**: Separates into vocals, drums, bass, and "other". In this project (Gen 1), "other" = guitar.
- **High Quality**: Uses htdemucs model for best separation quality
- **Batch Processing**: Can process multiple files efficiently
- **Error Handling**: Robust error handling and logging
- **Format Flexibility**: Supports both MP3 and WAV input/output

**Stems Generated**:
- `vocals.mp3` - Vocal tracks
- `drums.mp3` - Drum tracks  
- `bass.mp3` - Bass tracks
- `other.mp3` - Guitar tracks. 

**Output**: Organized directory structure with separated stems

### 3. Spectral Analysis (`spectral_analysis.py`)

**Purpose**: Converts audio stems into frequency-domain data using Short-Time Fourier Transforms (STFT).

**Key Features**:
- **STFT Processing**: Short-Time Fourier Transform with configurable parameters
- **Structured Data**: Frequency-domain information in organized Byte objects
- **Normalization**: Automatic normalization for neural network input
- **Frequency Analysis**: Energy analysis across different frequency ranges
- **Time Resolution**: High temporal resolution for real-time visualization

**Data Structure**:
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
- **Progress Tracking**: Detailed logging and progress reporting

**Output Format**:
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

### 5. Unity Visualization System

**Purpose**: Real-time 3D visualization of spectral data using modular Unity architecture.

**Key Components**:
- **AudioStrand**: Main coordinator class managing visualization systems
- **StrandMeshGenerator**: Handles mesh creation and deformation
- **StrandRenderer**: Manages visual rendering components
- **StrandPhysics**: Handles motion and physics calculations

**Architecture Principles**:
- **Separation of Concerns**: Each class has a single, well-defined responsibility
- **Component-Based Design**: Modular Unity components for flexibility
- **Data-Driven Visualization**: Visual parameters driven by spectral data
- **Performance Optimization**: Efficient rendering for real-time visualization

## Data Flow Architecture

### 1. Input Processing
```
Raw Audio → Segmentation → Dual Sampling Strategy
     ↓              ↓              ↓
  MP3/WAV → Symmetric (70%) → Regular intervals
     ↓              ↓              ↓
  Files → Random (30%) → Variable positions/lengths
```

### 2. Audio Analysis
```
Segmented Audio → Stem Separation → Spectral Analysis
      ↓                ↓                ↓
   WAV Files → Demucs Processing → STFT Conversion
      ↓                ↓                ↓
  Metadata → 4 Stems Each → Frequency Domain Data
```

### 3. Data Transformation
```
Spectral Data → JSON Serialization → Unity Consumption
      ↓                ↓                ↓
  Byte Objects → Structured Format → Real-time Access
      ↓                ↓                ↓
  Frequency Info → File Storage → 3D Visualization
```

## Configuration Management

### Path Configuration (`config/paths.py`)

**Purpose**: Centralized path management supporting both local and external storage.

**Key Features**:
- **Flexible Storage**: Supports local and external drive storage
- **Environment Variables**: Configurable via environment variables
- **Automatic Detection**: Detects available storage options
- **Fallback System**: Graceful degradation when external storage unavailable

**Configuration Options**:
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

## Performance Considerations

### Audio Processing
- **Segmentation**: Efficient audio slicing with minimal memory overhead
- **Stem Separation**: GPU-accelerated Demucs processing when available
- **Spectral Analysis**: Optimized STFT with configurable window sizes
- **Batch Processing**: Parallel processing capabilities for large datasets

### Unity Visualization
- **Component Optimization**: Efficient Unity component design
- **Memory Management**: Proper cleanup of dynamically created objects
- **Rendering Pipeline**: Optimized for real-time performance
- **Data Loading**: Efficient JSON parsing and data structure access

## Error Handling and Robustness

### Audio Processing Pipeline
- **File Validation**: Checks for valid audio files before processing
- **Error Recovery**: Continues processing even if individual files fail
- **Logging**: Comprehensive logging for debugging and monitoring
- **Data Integrity**: Validation of output data structures

### Unity Visualization
- **Null Safety**: Robust handling of missing or invalid data
- **Component Validation**: Automatic creation of required components
- **Resource Management**: Proper cleanup to prevent memory leaks
- **Error Reporting**: Clear error messages for debugging

## Extensibility and Modularity

### Python Pipeline
- **Modular Design**: Each processing stage is independent
- **Configurable Parameters**: Easy adjustment of processing parameters
- **Plugin Architecture**: Extensible for additional audio processing steps
- **Data Format Flexibility**: Support for various input/output formats

### Unity Visualization
- **Component-Based Architecture**: Easy to add new visualization components
- **ScriptableObject Configuration**: Flexible parameter management
- **Event System Integration**: Decoupled system communication
- **Custom Shaders**: Extensible rendering pipeline

## Learning Value

This architecture demonstrates:
- **Audio Signal Processing**: STFT, spectral analysis, stem separation
- **Data Pipeline Design**: Modular, configurable processing systems
- **Unity Development**: Component-based architecture, real-time rendering
- **System Integration**: Python-to-Unity data flow
- **Performance Optimization**: Efficient audio and visual processing
- **Software Engineering**: Separation of concerns, modular design, error handling

## Future Enhancements

### Planned Improvements
- **Real-time Audio Input**: Direct audio input for live visualization
- **Advanced Visualization**: Additional visual effects and rendering techniques
- **Machine Learning Integration**: AI-driven parameter generation
- **Multi-platform Support**: Cross-platform compatibility
- **Performance Optimization**: Further optimization for large datasets

### Extension Points
- **Additional Audio Analysis**: More sophisticated audio feature extraction
- **Custom Visualization Components**: User-defined visual effects
- **Network Integration**: Remote audio processing and visualization
- **Data Export**: Additional output formats for external analysis 