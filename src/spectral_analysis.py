import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import List, Dict, Optional
import soundfile as sf

@dataclass
class Byte:
    """
    A Byte represents spectral information for a segment of audio at a specific time.
    """
    time: float # time index of the byte
    stem_label: str # label of the stem (e.g. 'vocals', 'drums', 'bass', 'other')
    magnitudes: np.ndarray # array of magnitudes - corresponds to frequency bins
    phases: np.ndarray # phases of the byte - corresponds to frequency bins
    frequency_bins: np.ndarray # frequency bins of the byte
    sample_rate: int # sample rate of the audio from which the byte was created
    
    def __post_init__(self):
        """Validate and process data after initialization"""
        # Ensure arrays are numpy arrays
        self.magnitudes = np.array(self.magnitudes, dtype=np.float32)
        self.phases = np.array(self.phases, dtype=np.float32)
        self.frequency_bins = np.array(self.frequency_bins, dtype=np.float32)
        
        # Validate dimensions
        # magnitudes and phases must have same length - each magnitude corresponds to a phase
        assert len(self.magnitudes) == len(self.phases), "Magnitudes and phases must have same length"
        # magnitudes and frequency bins must have same length - each magnitude corresponds to a frequency bin
        assert len(self.magnitudes) == len(self.frequency_bins), "Frequency bins must match spectral data"

    def get_vector(self) -> np.ndarray:
        """Returns a normalized vector for neural network input"""
        # Normalize magnitudes to [0, 1] range 
        norm_magnitudes = self.magnitudes / np.max(self.magnitudes)
        # Normalize phases to [0, 1] range (from [-π, π]) 
        norm_phases = (self.phases + np.pi) / (2 * np.pi)
        
        return np.concatenate([norm_magnitudes, norm_phases])

    def get_spectral_info(self) -> Dict:
        """Returns detailed spectral information"""
        return {
            'time': self.time,
            'stem': self.stem_label,
            'magnitudes': self.magnitudes,
            'phases': self.phases,
            'frequencies': self.frequency_bins,
            'sample_rate': self.sample_rate
        }

    def get_frequency_bands(self) -> Dict[str, float]:
        """
        Analyze frequency bands (useful for visualization)
        Returns energy in different frequency bands
        """
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 20000)
        }
        
        energy = {}
        for band_name, (low, high) in bands.items():
            mask = (self.frequency_bins >= low) & (self.frequency_bins <= high)
            energy[band_name] = np.sum(self.magnitudes[mask])
            
        return energy

class ByteExtractor:
    """
    Extracts Bytes from audio stems using spectral analysis.

    Responsibilities:
   - Performing STFT on audio stems to create Bytes
   - Extracting magnitude and phase information from the STFT
   - Creating Byte objects for each time window in STFT
    """
    def __init__(self, 
                 window_size: int = 2048,
                 hop_length: int = 512,
                 sample_rate: int = 44100):
        """
        Initialize the ByteExtractor with the given parameters

        Args:
            window_size: Size of the STFT window (default 2048)
            hop_length: Number of samples between successive STFT columns (default 512)
            sample_rate: Sample rate of the audio (default 44100 Hz)
        """
        self.window_size = window_size # determines frequency resolution
        self.hop_length = hop_length # determines overlap between consecutive windows
        self.sample_rate = sample_rate # sample rate of the audio input
        self.window = signal.windows.hann(window_size) # window function for STFT (ideal for low spectral leakage)

    def extract_bytes(self, 
                     audio_data: np.ndarray, 
                     stem_label: str) -> List[Byte]:
        """
        Extract Bytes from an audio stem
        
        Args:
            audio_data: Audio time series (samples x channels) {Can be Mono or Stereo}
            stem_label: Label for the stem ('vocals', 'drums', etc.)
            
        Returns:
            List of Byte objects, one for each time window in the STFT
        """
        # Convert stereo to mono if needed (done by averaging the two channels)
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data

        # Compute STFT
        # f: Array of frequency bins centers
        # t: Array of time indices for each window
        # Zxx: STFT matrix of the audio data containing complex values (M x P)
        f, t, Zxx = signal.stft(audio_mono, 
                               fs=self.sample_rate,
                               window=self.window,
                               nperseg=self.window_size,
                               noverlap=self.window_size - self.hop_length)

        # Extract magnitude and phase from STFT matrix
        magnitudes = np.abs(Zxx) # amplitude of each frequency component 
        phases = np.angle(Zxx) # phase of each frequency componentn

        # Create Bytes for each time window
        bytes_list = []
        for i in range(len(t)):
            byte = Byte(
                time=t[i],
                stem_label=stem_label,
                magnitudes=magnitudes[:, i],
                phases=phases[:, i],
                frequency_bins=f,
                sample_rate=self.sample_rate
            )
            # Append the byte to the list of bytes
            bytes_list.append(byte)

        return bytes_list

    def process_stem(self, 
                    audio_path: str, 
                    stem_label: str) -> List[Byte]:
        """
        Process a single audio stem file

        Args:
            audio_path: Path to the audio file
            stem_label: Label for the stem ('vocals', 'drums', etc.)

        Returns:
            List of Byte objects, one for each time window in the STFT
                - Contain stem spectral data
        """
        audio_data, sr = sf.read(audio_path)
        assert sr == self.sample_rate, f"Expected {self.sample_rate}Hz, got {sr}Hz"
        return self.extract_bytes(audio_data, stem_label)
