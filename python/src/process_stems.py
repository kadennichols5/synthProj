import os
import json
import numpy as np
from spectral_analysis import ByteExtractor
from typing import Dict, List
import soundfile as sf
from pydub import AudioSegment

def process_stem_file(stem_path: str, stem_label: str, extractor: ByteExtractor) -> List[Dict]:
    """
    Process a single stem file and extract spectral information.
    
    Args:
        stem_path: Path to the stem file
        stem_label: Label of the stem (vocals, drums, bass, other)
        extractor: ByteExtractor instance
        
    Returns:
        List of dictionaries containing spectral information
    """
    try:
        # Read the audio file (handling both MP3 and WAV)
        if stem_path.endswith('.mp3'):
            # Convert MP3 to WAV for processing
            audio = AudioSegment.from_mp3(stem_path)
            # Convert to numpy array
            audio_data = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                audio_data = audio_data.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
            sr = audio.frame_rate
        else:
            audio_data, sr = sf.read(stem_path)
        
        # Extract bytes
        bytes_list = extractor.extract_bytes(audio_data, stem_label)
        
        # Convert bytes to serializable format
        spectral_data = []
        for byte in bytes_list:
            spectral_data.append({
                'time': float(byte.time),
                'stem': byte.stem_label,
                'magnitudes': byte.magnitudes.tolist(),
                'phases': byte.phases.tolist(),
                'frequencies': byte.frequency_bins.tolist(),
                'sample_rate': byte.sample_rate
            })
        
        return spectral_data
    except Exception as e:
        print(f"Error processing {stem_path}: {str(e)}")
        return []

def process_segmented_stems(input_dir: str, output_dir: str):
    """
    Process all segmented stems and save spectral information.
    
    Args:
        input_dir: Directory containing segmented stems
        output_dir: Directory to save spectral data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ByteExtractor
    extractor = ByteExtractor()
    
    # Process each segment directory
    for segment_dir in os.listdir(input_dir):
        segment_path = os.path.join(input_dir, segment_dir)
        if not os.path.isdir(segment_path):
            continue
            
        print(f"Processing segment: {segment_dir}")
        
        # Dictionary to store spectral data for all stems
        segment_data = {
            'segment_name': segment_dir,
            'stems': {}
        }
        
        # Look for stems in the htdemucs subdirectory
        htdemucs_path = os.path.join(segment_path, 'htdemucs', segment_dir)
        if not os.path.exists(htdemucs_path):
            print(f"Warning: No htdemucs directory found for {segment_dir}")
            continue
        
        # Process each stem
        for stem_file in os.listdir(htdemucs_path):
            if not stem_file.endswith('.mp3'):
                continue
                
            stem_path = os.path.join(htdemucs_path, stem_file)
            stem_label = os.path.splitext(stem_file)[0]  # Remove .mp3 extension
            
            # Extract spectral information
            spectral_data = process_stem_file(stem_path, stem_label, extractor)
            if spectral_data:
                segment_data['stems'][stem_label] = spectral_data
        
        # Save spectral data for this segment
        output_file = os.path.join(output_dir, f"{segment_dir}_spectral.json")
        with open(output_file, 'w') as f:
            json.dump(segment_data, f, indent=2)
        
        print(f"Saved spectral data to {output_file}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/kadensnichols/Desktop/synthProj/segmented_stems"
    output_dir = "/Users/kadensnichols/Desktop/synthProj/spectral_data"
    
    # Process all segmented stems
    process_segmented_stems(input_dir, output_dir) 