import os
import json
import numpy as np
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from paths import get_segmented_audio_path
from pydub import AudioSegment

"""
This file takes .mp3 files and samples them. 
70% of samples will have symmetric overlap. 
30% of samples randomly taken
"""

def extract_segments(audio_path, segment_len_ms = 5000, stride_ms = 2000, random_ratio = 0.3, random_len_range = (3000, 7000), output_dir = "segments_out"):
    os.makedirs(output_dir, exist_ok = True)
    audio = AudioSegment.from_file(audio_path)
    audio_duration_ms = len(audio)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    metadata = []
    segment_id = 0
    
    #building symmetrically sampled segments
    sym_segments = []
    for start in range(0, audio_duration_ms - segment_len_ms + 1, stride_ms):
        end = start + segment_len_ms
        segment = audio[start:end]
        out_path = os.path.join(output_dir, f"{base_name}_sym_{segment_id:04d}.wav")
        segment.export(out_path, format = "wav")
        metadata.append({
            "filename": out_path,
            "type": "symmetric",
            "end_ms": end,
            "duration_ms": segment_len_ms
            })
        sym_segments.append((start, end))
        segment_id += 1
    
    #building randomly generated segments
    num_random_segments = int(len(sym_segments) * random_ratio)
    used_windows = set()

    for n in range(num_random_segments):
        max_attempts = 10
        for attempt in range(max_attempts):
            rand_len = random.randint(*random_len_range)
            # pick a random start so the segment fits in the audio
            start = random.randint(0, audio_duration_ms - rand_len)
            end = start + rand_len

            # avoiding too much overlap of existing symmetric segments
            if not any(abs(start - sym_start) < 600 for (sym_start, _) in sym_segments):
                segment = audio[start:end]
                out_path = os.path.join(output_dir, f"{base_name}_rndm_{segment_id:04d}.wav")
                segment.export(out_path, format='wav')
                metadata.append({
                    "filename": out_path,
                    "type": "random",
                    "start_ms": start,
                    "end_ms": end,
                    "duration_ms": rand_len
                })
                segment_id += 1
                break

    # save metadata
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {segment_id} segments and metadata to {output_dir}")

# --- Process all files in the input directory ---

if __name__ == "__main__":
    output_dir = get_segmented_audio_path()
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "/Users/kadensnichols/Desktop/synthProj/BassVocalDrumGuitar"
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3'))]

    for filename in audio_files:
        audio_path = os.path.join(input_dir, filename)
        extract_segments(audio_path, output_dir=output_dir)



