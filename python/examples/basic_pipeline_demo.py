#!/usr/bin/env python3
"""
Basic Pipeline Demo

This script demonstrates the complete audio processing pipeline
using a single audio file. It shows how to:

1. Segment an audio file
2. Separate the segments into stems
3. Perform spectral analysis
4. View the results

Usage:
    python basic_pipeline_demo.py [audio_file_path]
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from segment_generator import extract_segments
from audio_separation import AudioSeparator
from spectral_analysis import ByteExtractor
from process_stems import process_stem_file

def demo_pipeline(audio_file_path):
    """
    Run a complete demo of the audio processing pipeline.
    
    Args:
        audio_file_path: Path to the input audio file
    """
    print(f"🎵 Audio Processing Pipeline Demo")
    print(f"📁 Input file: {audio_file_path}")
    print("=" * 50)
    
    # Create temporary directories for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        segmented_dir = os.path.join(temp_dir, "segmented")
        stems_dir = os.path.join(temp_dir, "stems")
        spectral_dir = os.path.join(temp_dir, "spectral")
        
        os.makedirs(segmented_dir, exist_ok=True)
        os.makedirs(stems_dir, exist_ok=True)
        os.makedirs(spectral_dir, exist_ok=True)
        
        # Step 1: Audio Segmentation
        print("\n🔪 Step 1: Audio Segmentation")
        print("Chopping audio into smaller segments...")
        
        try:
            extract_segments(
                audio_file_path,
                segment_len_ms=3000,  # 3 second segments for demo
                stride_ms=1500,       # 1.5 second overlap
                random_ratio=0.2,     # 20% random segments
                random_len_range=(2000, 4000),
                output_dir=segmented_dir
            )
            
            # Count generated segments
            wav_files = [f for f in os.listdir(segmented_dir) if f.endswith('.wav')]
            print(f"✅ Generated {len(wav_files)} audio segments")
            
        except Exception as e:
            print(f"❌ Segmentation failed: {str(e)}")
            return
        
        # Step 2: Stem Separation
        print("\n🎼 Step 2: Stem Separation")
        print("Separating audio into individual stems...")
        
        try:
            # Process first segment as demo
            if wav_files:
                first_segment = wav_files[0]
                segment_path = os.path.join(segmented_dir, first_segment)
                segment_name = os.path.splitext(first_segment)[0]
                
                separator = AudioSeparator(segment_path, stems_dir)
                separator.separate_audio()
                
                print(f"✅ Separated stems for: {first_segment}")
                
                # Check what stems were generated
                htdemucs_path = os.path.join(stems_dir, 'htdemucs', segment_name)
                if os.path.exists(htdemucs_path):
                    stems = [f for f in os.listdir(htdemucs_path) if f.endswith('.mp3')]
                    print(f"   Generated stems: {', '.join(stems)}")
                
        except Exception as e:
            print(f"❌ Stem separation failed: {str(e)}")
            return
        
        # Step 3: Spectral Analysis
        print("\n📊 Step 3: Spectral Analysis")
        print("Converting audio to spectral data...")
        
        try:
            # Initialize ByteExtractor
            extractor = ByteExtractor()
            
            # Process one stem as demo
            if os.path.exists(htdemucs_path):
                stem_files = [f for f in os.listdir(htdemucs_path) if f.endswith('.mp3')]
                if stem_files:
                    # Process vocals stem as example
                    vocals_stem = os.path.join(htdemucs_path, 'vocals.mp3')
                    if os.path.exists(vocals_stem):
                        spectral_data = process_stem_file(vocals_stem, "vocals", extractor)
                        
                        print(f"✅ Generated {len(spectral_data)} spectral bytes")
                        
                        # Show sample data
                        if spectral_data:
                            sample_byte = spectral_data[0]
                            print(f"   Sample byte:")
                            print(f"     Time: {sample_byte['time']:.2f}s")
                            print(f"     Stem: {sample_byte['stem']}")
                            print(f"     Magnitudes: {len(sample_byte['magnitudes'])} frequency bins")
                            print(f"     Phases: {len(sample_byte['phases'])} frequency bins")
                            print(f"     Sample rate: {sample_byte['sample_rate']} Hz")
                            
                            # Save sample data
                            output_file = os.path.join(spectral_dir, f"{segment_name}_vocals_spectral.json")
                            with open(output_file, 'w') as f:
                                json.dump({
                                    'segment_name': segment_name,
                                    'stems': {
                                        'vocals': spectral_data
                                    }
                                }, f, indent=2)
                            
                            print(f"   💾 Saved spectral data to: {output_file}")
        
        except Exception as e:
            print(f"❌ Spectral analysis failed: {str(e)}")
            return
        
        print("\n🎉 Pipeline Demo Completed Successfully!")
        print("=" * 50)
        print("📋 Summary:")
        print(f"   • Segmented audio into {len(wav_files)} chunks")
        print(f"   • Separated stems for 1 segment")
        print(f"   • Generated spectral data for vocals stem")
        print(f"   • All data saved to temporary directory")

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python basic_pipeline_demo.py <audio_file_path>")
        print("\nExample:")
        print("  python basic_pipeline_demo.py /path/to/song.mp3")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        sys.exit(1)
    
    # Check file format
    if not audio_file_path.lower().endswith(('.mp3', '.wav')):
        print(f"❌ Error: Unsupported file format. Please use .mp3 or .wav files")
        sys.exit(1)
    
    try:
        demo_pipeline(audio_file_path)
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 