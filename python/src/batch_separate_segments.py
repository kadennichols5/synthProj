import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from paths import get_segmented_audio_path, get_segmented_stems_path
from audio_separation import AudioSeparator

segmented_dir = get_segmented_audio_path()
output_dir = get_segmented_stems_path()
os.makedirs(output_dir, exist_ok=True)

# Log file to keep track of processed files
log_file = "separated_files.txt"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        completed = set(line.strip() for line in f)
else:
    completed = set()

# Get all .wav files in segmented_audio
all_files = [f for f in os.listdir(segmented_dir) if f.endswith('.wav')]

for wav_file in all_files:
    if wav_file in completed:
        print(f"Already processed: {wav_file}")
        continue

    input_path = os.path.join(segmented_dir, wav_file)
    song_output_dir = os.path.join(output_dir, os.path.splitext(wav_file)[0])
    separator = AudioSeparator(input_path, song_output_dir)
    try:
        separator.separate_audio()
        print(f"Processed: {wav_file}")
        with open(log_file, "a") as f:
            f.write(wav_file + "\n")
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        with open("separation_errors.txt", "a") as ef:
            ef.write(f"{wav_file}: {e}\n")
