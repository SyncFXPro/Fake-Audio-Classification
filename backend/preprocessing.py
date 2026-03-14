import librosa
import numpy as np
import torch
#this is the "production" code for the spectrogram, not the  #load_audio.py code..
def audio_to_spectrogram(audio_path, sr=16000, duration=2.0, n_fft=1024, hop_length=256, n_mels=128):
    """
    Convert audio file to normalized Log-Mel spectrogram.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate (default 16kHz)
        duration: Duration in seconds (default 2.0)
        n_fft: FFT window size (default 1024)
        hop_length: Hop length for STFT (default 256)
        n_mels: Number of mel bins (default 128)
    
    Returns:
        normalized: Normalized log-mel spectrogram, shape [128, ~200]
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        normalized = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8) #Making sure the spectrogram makes sense.
        return normalized
    
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        raise


def audio_to_tensor(audio_path):
    """
    Convert audio file to PyTorch tensor ready for model input.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        tensor: Shape [1, 1, 128, time_frames] ready for model
    """
    spec = audio_to_spectrogram(audio_path)
    tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)
    return tensor


def batch_audio_to_spectrograms(audio_paths):
    """
    Convert multiple audio files to spectrograms.
    
    Args:
        audio_paths: List of audio file paths
    
    Returns:
        spectrograms: List of normalized spectrograms
    """
    spectrograms = []
    for path in audio_paths:
        try:
            spec = audio_to_spectrogram(path)
            spectrograms.append(spec)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    
    return spectrograms


if __name__ == "__main__":
    import os
    
    test_audio = r"H:\FAC\Dataset\for-2sec\for-2seconds\testing\fake\file2.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"
    
    if os.path.exists(test_audio):
        print("Testing audio_to_spectrogram...")
        spec = audio_to_spectrogram(test_audio)
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Mean: {spec.mean():.4f}, Std: {spec.std():.4f}")
        
        print("\nTesting audio_to_tensor...")
        tensor = audio_to_tensor(test_audio)
        print(f"Tensor shape: {tensor.shape}")
        
        print("\nPreprocessing module working correctly!")
    else:
        print(f"Test file not found: {test_audio}")
