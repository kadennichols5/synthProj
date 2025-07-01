"""
Audio Processing Pipeline - Main Entry Point

This script demonstrates the complete pipeline for processing audio files:
1. Loading and segmenting audio files
2. Separating audio into stems (vocals, drums, bass, other)
3. Performing spectral analysis to generate "Bytes"
4. Saving results for further processing

Usage:
    python main.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--mode MODE]

Modes:
    - 'full': Complete pipeline (segment -> separate -> spectral analysis)
    - 'segment': Only segment audio files
    - 'separate': Only separate stems (requires segmented audio)
    - 'spectral': Only spectral analysis (requires separated stems)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from paths import get_segmented_audio_path, get_segmented_stems_path, get_spectral_data_path

# Import our processing modules
from segment_generator import extract_segments
from audio_separation import AudioSeparator
from spectral_analysis import ByteExtractor
from process_stems import process_segmented_stems

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    Complete audio processing pipeline that handles the full workflow
    from raw audio files to spectral analysis results.
    """
    
    def __init__(self, input_dir=None, output_dir=None):
        """
        Initialize the pipeline with input and output directories.
        
        Args:
            input_dir: Directory containing raw audio files
            output_dir: Base directory for all outputs
        """
        self.input_dir = input_dir or "/Users/kadensnichols/Desktop/synthProj/BassVocalDrumGuitar"
        self.output_dir = output_dir or "/Users/kadensnichols/Desktop/synthProj"
        
        # Configure paths
        self.segmented_audio_path = get_segmented_audio_path()
        self.segmented_stems_path = get_segmented_stems_path()
        self.spectral_data_path = get_spectral_data_path()
        
        # Create output directories
        os.makedirs(self.segmented_audio_path, exist_ok=True)
        os.makedirs(self.segmented_stems_path, exist_ok=True)
        os.makedirs(self.spectral_data_path, exist_ok=True)
        
        logger.info(f"Pipeline initialized with:")
        logger.info(f"  Input directory: {self.input_dir}")
        logger.info(f"  Segmented audio: {self.segmented_audio_path}")
        logger.info(f"  Segmented stems: {self.segmented_stems_path}")
        logger.info(f"  Spectral data: {self.spectral_data_path}")
    
    def segment_audio_files(self):
        """
        Step 1: Segment raw audio files into smaller chunks.
        """
        logger.info("Starting audio segmentation...")
        
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        audio_files = [f for f in os.listdir(self.input_dir) 
                      if f.endswith(('.wav', '.mp3'))]
        
        if not audio_files:
            logger.warning(f"No audio files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        for filename in audio_files:
            logger.info(f"Segmenting: {filename}")
            audio_path = os.path.join(self.input_dir, filename)
            
            try:
                extract_segments(
                    audio_path, 
                    segment_len_ms=5000,  # 5 second segments
                    stride_ms=2000,       # 2 second overlap
                    random_ratio=0.3,     # 30% random segments
                    random_len_range=(3000, 7000),
                    output_dir=self.segmented_audio_path
                )
                logger.info(f"Successfully segmented: {filename}")
            except Exception as e:
                logger.error(f"Error segmenting {filename}: {str(e)}")
        
        logger.info("Audio segmentation complete!")
    
    def separate_stems(self):
        """
        Step 2: Separate segmented audio files into stems.
        """
        logger.info("Starting stem separation...")
        
        if not os.path.exists(self.segmented_audio_path):
            raise FileNotFoundError(f"Segmented audio directory not found: {self.segmented_audio_path}")
        
        # Get all WAV files in segmented audio directory
        wav_files = [f for f in os.listdir(self.segmented_audio_path) 
                    if f.endswith('.wav')]
        
        if not wav_files:
            logger.warning(f"No WAV files found in {self.segmented_audio_path}")
            return
        
        logger.info(f"Found {len(wav_files)} segments to separate")
        
        # Track processed files
        log_file = "separated_files.txt"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                completed = set(line.strip() for line in f)
        else:
            completed = set()
        
        for wav_file in wav_files:
            if wav_file in completed:
                logger.info(f"Already processed: {wav_file}")
                continue
            
            logger.info(f"Separating stems for: {wav_file}")
            input_path = os.path.join(self.segmented_audio_path, wav_file)
            song_output_dir = os.path.join(self.segmented_stems_path, 
                                         os.path.splitext(wav_file)[0])
            
            try:
                separator = AudioSeparator(input_path, song_output_dir)
                separator.separate_audio()
                
                # Log successful processing
                with open(log_file, "a") as f:
                    f.write(wav_file + "\n")
                
                logger.info(f"Successfully separated: {wav_file}")
            except Exception as e:
                logger.error(f"Error processing {wav_file}: {str(e)}")
                with open("separation_errors.txt", "a") as ef:
                    ef.write(f"{wav_file}: {str(e)}\n")
        
        logger.info("Stem separation complete!")
    
    def generate_spectral_data(self):
        """
        Step 3: Generate spectral data from separated stems.
        """
        logger.info("Starting spectral analysis...")
        
        if not os.path.exists(self.segmented_stems_path):
            raise FileNotFoundError(f"Segmented stems directory not found: {self.segmented_stems_path}")
        
        try:
            process_segmented_stems(self.segmented_stems_path, self.spectral_data_path)
            logger.info("Spectral analysis complete!")
        except Exception as e:
            logger.error(f"Error during spectral analysis: {str(e)}")
            raise
    
    def run_full_pipeline(self):
        """
        Run the complete audio processing pipeline.
        """
        logger.info("Starting full audio processing pipeline...")
        
        try:
            # Step 1: Segment audio files
            self.segment_audio_files()
            
            # Step 2: Separate stems
            self.separate_stems()
            
            # Step 3: Generate spectral data
            self.generate_spectral_data()
            
            logger.info("Full pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def demo_spectral_analysis(self, audio_file_path):
        """
        Demonstrate spectral analysis on a single audio file.
        
        Args:
            audio_file_path: Path to an audio file for demonstration
        """
        logger.info(f"Demonstrating spectral analysis on: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        # Initialize ByteExtractor
        extractor = ByteExtractor()
        
        # Process the file
        from process_stems import process_stem_file
        spectral_data = process_stem_file(audio_file_path, "demo", extractor)
        
        logger.info(f"Generated {len(spectral_data)} spectral bytes")
        
        # Show sample data
        if spectral_data:
            sample_byte = spectral_data[0]
            logger.info(f"Sample byte structure:")
            logger.info(f"  Time: {sample_byte['time']}")
            logger.info(f"  Stem: {sample_byte['stem']}")
            logger.info(f"  Magnitudes shape: {len(sample_byte['magnitudes'])}")
            logger.info(f"  Phases shape: {len(sample_byte['phases'])}")
            logger.info(f"  Frequencies shape: {len(sample_byte['frequencies'])}")
            logger.info(f"  Sample rate: {sample_byte['sample_rate']}")
        
        return spectral_data

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("--input_dir", type=str, 
                       help="Directory containing raw audio files")
    parser.add_argument("--output_dir", type=str,
                       help="Base directory for outputs")
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "segment", "separate", "spectral", "demo"],
                       help="Processing mode")
    parser.add_argument("--demo_file", type=str,
                       help="Audio file for spectral analysis demo")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AudioProcessingPipeline(args.input_dir, args.output_dir)
    
    try:
        if args.mode == "full":
            pipeline.run_full_pipeline()
        elif args.mode == "segment":
            pipeline.segment_audio_files()
        elif args.mode == "separate":
            pipeline.separate_stems()
        elif args.mode == "spectral":
            pipeline.generate_spectral_data()
        elif args.mode == "demo":
            if not args.demo_file:
                logger.error("Demo mode requires --demo_file argument")
                return
            pipeline.demo_spectral_analysis(args.demo_file)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()