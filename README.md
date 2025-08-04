# Synesthetic Audio Visualization System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Unity](https://img.shields.io/badge/Unity-6.0.0.31f1-000000.svg)](https://unity.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey.svg)](https://apple.com)

A comprehensive system that transforms music into dynamic 3D visualizations through advanced audio processing and real-time Unity rendering.

## 🎯 Project Overview

This project demonstrates full-stack development combining:
- **Audio Processing Pipeline** (Python) - Converts music into spectral data
- **Real-time Visualization** (Unity) - Renders dynamic 3D visual elements

The system processes audio through spectral analysis and creates responsive 3D visualizations that react to music in real-time, showcasing advanced audio processing, data transformation, and interactive visualization techniques.

## 🏗️ Architecture

```
Music Files → Python Pipeline → Spectral Data → Unity Visualization
     ↓              ↓                ↓              ↓
  Raw Audio → Stem Separation → Frequency Data → 3D Visuals
```

### Components

- **[Python Audio Pipeline](./python/)** - Audio processing and spectral analysis
- **[Unity Visualization](./unityViz/)** - Real-time 3D visual rendering
- **[Project Architecture](./ARCHITECTURE.md)** - Detailed system documentation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Unity 6.0.0.31f1 (macOS)
- Audio files (MP3/WAV format)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/synthProj.git
   cd synthProj
   ```

2. **Setup Python Pipeline**
   ```bash
   cd python
   pip install -r requirements.txt
   pip install demucs  # For audio separation
   ```

3. **Setup Unity Project**
   - Open `unityViz/SynthAIViz/` in Unity 6.0.0.31f1
   - Import required packages (see [Unity README](./unityViz/README.md))

### Basic Usage

1. **Process Audio Files**
   ```bash
   cd python
   python src/main.py --mode full --input /path/to/music
   ```

2. **Run Unity Visualization**
   - Open the Unity project
   - Load spectral data files
   - Press Play to see real-time visualization

## 📁 Project Structure

```
synthProj/
├── python/                    # Audio processing pipeline
│   ├── src/                   # Core processing scripts
│   ├── config/                # Configuration and paths
│   ├── examples/              # Usage examples
│   └── README.md              # Python component documentation
├── unityViz/                  # Unity visualization project
│   └── SynthAIViz/           # Unity project files
│       ├── Assets/Scripts/    # C# visualization scripts
│       └── README.md          # Unity component documentation
├── segmented_audio/           # Generated audio segments
├── segmented_stems/           # Separated audio components
├── spectral_data/             # Spectral analysis results
├── ARCHITECTURE.md            # Detailed system architecture
└── README.md                  # This file
```

## 🔧 Key Features

### Python Pipeline
- **Audio Segmentation** - Breaks tracks into manageable segments
- **Stem Separation** - Isolates vocals, drums, bass, and other instruments
- **Spectral Analysis** - Converts audio to frequency-domain data
- **Batch Processing** - Handles large audio collections efficiently

### Unity Visualization
- **Modular Architecture** - Component-based design for maintainability
- **Real-time Rendering** - Dynamic 3D visual elements
- **Audio Responsiveness** - Visuals react to spectral data changes
- **Performance Optimized** - Efficient rendering for smooth visualization

## 🎨 Visualization Examples

The Unity component creates various visual effects:
- **Strand Visualization** - Dynamic audio trails
- **Particle Systems** - Responsive particle effects
- **Mesh Deformation** - Real-time geometry changes
- **Color Transitions** - Smooth visual parameter changes

## 📊 Data Flow

1. **Input**: Raw audio files (MP3/WAV)
2. **Segmentation**: Audio broken into segments with two sampling strategies:
   - **70% Symmetric**: Regular stride-based sampling (5-second segments, 2-second overlap)
   - **30% Random**: Random start positions and lengths (3-7 second segments)
3. **Separation**: Each segment split into 4 stems (vocals, drums, bass, other)
4. **Analysis**: STFT spectral analysis on each stem
5. **Output**: JSON files with frequency-domain data
6. **Visualization**: Unity reads JSON data for real-time rendering

## 🛠️ Configuration

The system supports flexible configuration:
- **Local Storage**: Default local file storage
- **External Storage**: Configurable external drive support
- **Environment Variables**: Customizable paths and settings
- **Generation Versioning**: Support for multiple data versions

## 📚 Documentation

- **[Python Pipeline](./python/README.md)** - Detailed audio processing documentation
- **[Unity Visualization](./unityViz/README.md)** - Visualization component guide
- **[System Architecture](./ARCHITECTURE.md)** - Complete technical architecture

## 🎓 Learning Value

This project demonstrates:
- **Audio Signal Processing** - STFT, spectral analysis, stem separation
- **Data Pipeline Design** - Modular, configurable processing systems
- **Unity Development** - Component-based architecture, real-time rendering
- **System Integration** - Python-to-Unity data flow
- **Performance Optimization** - Efficient audio and visual processing

## 🤝 Contributing

This is a portfolio project demonstrating full-stack audio visualization development. The modular architecture makes it easy to extend and modify individual components.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for shapes and sounds**



