import os
from segment_generator import extract_segments

song_directory = "/Users/kadensnichols/Desktop/synthProj/BassVocalDrumGuitar"
songs = [f for f in os.listdir(song_directory) if os.path.isfile(os.path.join(directory, f))]
print(songs)

cleaned_songs = [s.replace(".mp3.mps", ".mp3") for s in songs]



