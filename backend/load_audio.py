import librosa
import os
#Deprecated!
audio_path =  r"H:\FAC\public\audio\freepik__elegant-digital-chime_-soft-marimba-note_-airy-pluck_-non-aggressive_-friendly-in__1740 (1).mp3"


def load_audio_file(file_path):
    """
    Loads an audio file and returns a list of audio samples.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not file_path.endswith(".mp3"):
        raise ValueError("Invalid file type. Please provide a .mp3 file.")
    try: 
        print("Loading audio file:", file_path)
        audio_file = librosa.load(file_path)
        audio_list = list(audio_file)
        return audio_list
    except Exception as e:
        print("Error loading audio file:", e)
        pass


def main():
    """
    Main function to load the audio file.
    """
    try:
        load_audio_file(audio_path)
    except Exception as e:
        print("Error loading audio file:", e)
        pass
    pass

if __name__ == "__main__":
    main()