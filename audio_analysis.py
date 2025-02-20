"""
This file will contain the core functions for loading and analyzing audio files.

Functions to load audio files (e.g., using Librosa or SciPy) [Phase 1].
◦
Functions to perform spectral analysis (e.g., FFT to obtain frequency, amplitude, and phase data) .
◦
Functions to structure the spectral data into "Bytes" (B(t, S)). 
This may involve creating data structures (e.g., NumPy arrays or lists) to hold the 
spectral information for each byte.


take in audio file, seperate into stem, deconstruct into spectral data, map into bytes
"""

import os
import subprocess
import numpy as np
import time


class AudioSeperator:
    def __init__(self, input_path, output_path, demucs_options = None):
        """
        Initializes the AudioSeparator class.
        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to the output directory.
            demucs_options (dict, optional): Dictionary of Demucs options. Defaults to None.
        """
        
        self.input_path = input_path # audio file path to separate
        self.output_path = output_path # path to save separated stems
        self.demucs_options = demucs_options if demucs_options else {}

def set_demucs_options(self, options):
    """
    Sets or updates the Demucs options.

    Args:
        options (dict): Dictionary of Demucs options. 
                        Possible keys include:
                            - model_name (str): Name of the Demucs model to use.
                            - two_stems (str): Isolate two stems (e.g., 'vocals').
                            - mp3 (bool): Save output as MP3.
                            - mp3_bitrate (str): MP3 bitrate (e.g., '320').
                            - shifts (int): Number of random shifts for averaging.
                            - overlap (float): Overlap between windows (0 to 1).
                            - cpu (bool): Use CPU instead of GPU.
                            - segment (int): Segment length for memory efficiency.
    """
    # Validate input type
    if not isinstance(options, dict):
        raise ValueError("options must be a dictionary.")

    # Update Demucs options
    self.demucs_options.update(options)


def separate_audio(self):
    """
    Separates audio file into stems using Demucs.

    This function leverages the Demucs model to separate an audio file into individual stems, such as vocals, drums, bass, and others.
    It supports various customization options, including:
        - Model Selection: Choose the Demucs model variant (e.g., 'htdemucs', 'htdemucs_ft').
        - Two Stems: Isolate only two stems (e.g., vocals and accompaniment).
        - MP3 Output: Save the output as MP3 with customizable bitrate.
        - Shifts: Number of random shifts for prediction averaging.
        - Overlap: Amount of overlap between prediction windows.
        - CPU Mode: Option to use CPU instead of GPU.
        - Segment Length: Specify segment length to reduce memory usage.

    Attributes:
        model_name (str): Name of the Demucs model to use.
        two_stems (str, optional): If specified, isolates two stems (e.g., 'vocals').
        mp3 (bool): If True, saves the output as MP3.
        mp3_bitrate (str): MP3 bitrate in kbps (e.g., '320').
        shifts (int, optional): Number of random shifts for prediction averaging.
        overlap (float, optional): Overlap between prediction windows (default: 0.25).
        cpu (bool): If True, uses CPU for processing.
        segment (int, optional): Segment length in seconds to reduce memory usage.
        input_path (str): Path to the input audio file.
        output_path (str): Path to the output directory.

    """
    try:
        # Ensure the output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # Construct the Demucs command
        command = ['demucs', '-n', self.model_name]

        # Add optional arguments
        if self.two_stems:
            command.extend(['--two-stems', self.two_stems])
        if self.mp3:
            command.append('--mp3')
            command.extend(['--mp3-bitrate', self.mp3_bitrate])
        if self.shifts is not None:
            command.extend(['--shifts', str(self.shifts)])
        if self.overlap is not None:
            command.extend(['--overlap', str(self.overlap)])
        if self.cpu:
            command.extend(['-d', 'cpu'])
        if self.segment is not None:
            command.extend(['--segment', str(self.segment)])

        # Add input and output paths
        command.append(self.input_path)
        command.extend(['-o', self.output_path])

        # Execute the Demucs command
        print(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Demucs separation complete.")

    except subprocess.CalledProcessError as e:
        print(f"Error during Demucs processing: {e.stderr}")
    except FileNotFoundError:
        print("Error: File not found. Check to make sure file path exists.")


        


