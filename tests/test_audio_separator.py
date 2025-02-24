import pytest
import os
import sys
import numpy as np

# Debug prints
print("Current directory:", os.getcwd())
print("Parent directory:", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Python path:", sys.path)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from audio_analysis import AudioSeparator
print("Successfully imported AudioSeparator")


class TestAudioSeparator:

    @pytest.fixture
    def test_tracks(self):
        raw_audio_dir = os.path.join(parent_dir, "RawAudioFiles")
        tracks = [
            os.path.join(raw_audio_dir, f"test_track{i}.mp3") 
            for i in range(1, 4)
            ]
        return tracks
    
    @pytest.fixture
    def test_files_dir(self):
        # create a test files directory
        test_dir = "tests/test_files"
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    @pytest.fixture
    def audio_separator(self, test_tracks):
        input_path = test_tracks[0]
        output_path = os.path.join(parent_dir, "separated_output")
        return AudioSeparator(input_path=input_path, output_path=output_path)
    
    def test_initialization(self, audio_separator):
        """Test if AudioSeparator initializes with correct attributes"""
        assert isinstance(audio_separator, AudioSeparator)
        assert audio_separator.input_path.endswith("test_track1.mp3")
        assert "separated_output" in audio_separator.output_path
        assert isinstance(audio_separator.demucs_options, dict)
        assert len(audio_separator.demucs_options) == 0

    def test_stem_separation(self, audio_separator):
        """Test if stems are correctly separated"""
        try:
            stems = audio_separator.separate_audio()
            
            # Check if stems dictionary contains expected keys
            expected_stems = ['vocals', 'drums', 'bass', 'other']
            for stem in expected_stems:
                assert stem in stems, f"Missing {stem} stem"
                assert isinstance(stems[stem], np.ndarray), f"{stem} stem should be a numpy array"
                assert stems[stem].shape[0] == 2, f"{stem} stem should have 2 channels"
        
        except Exception as e:
            pytest.fail(f"Separation failed with error: {str(e)}")

    def test_demucs_options(self, audio_separator):
        """Test setting Demucs options"""
        test_options = {
            "model": "htdemucs",
            "two_stems": "vocals",
            "mp3": True
        }
        audio_separator.set_demucs_options(test_options)
        assert audio_separator.demucs_options == test_options

    def test_invalid_file(self):
        """Test handling of non-existent input file"""
        with pytest.raises(FileNotFoundError):
            separator = AudioSeparator("nonexistent.mp3", "output")
            separator.separate_audio()

    def test_batch_separation(self, test_tracks):
        """Test separation of multiple tracks"""
        # Get the directory containing test tracks
        input_dir = os.path.dirname(test_tracks[0])
        
        separator = AudioSeparator(
            input_path="",  # Will be set per file
            output_path="batch_separated_output"
        )
        
        try:
            results = separator.batch_separate_audio(input_dir)
            
            # Verify results
            assert len(results) > 0, "No tracks were processed"
            
            # Check if all test tracks were processed
            expected_files = [os.path.basename(track) for track in test_tracks]
            for file_name in expected_files:
                assert file_name in results, f"Missing results for {file_name}"
                
            # Check output directory structure
            for track_name in results:
                base_name = os.path.splitext(track_name)[0]
                model_dir = os.path.join("batch_separated_output", "mdx_extra_q", base_name)
                
                # Check for stem files
                for stem in ['vocals', 'drums', 'bass', 'other']:
                    stem_path = os.path.join(model_dir, f"{stem}.mp3")
                    assert os.path.exists(stem_path), f"Missing {stem} for {track_name}"
                    assert os.path.getsize(stem_path) > 0, f"Empty {stem} file for {track_name}"
                    
        except Exception as e:
            pytest.fail(f"Batch separation failed with error: {str(e)}")