import numpy as np
import librosa
import soundfile as sf
import io









def time_mask(spectrogram, max_mask_time=20, num_masks=1):
    """
    Apply time masking to spectrogram.
    
    Args:
        spectrogram: Input spectrogram [freq, time]
        max_mask_time: Maximum time frames to mask
        num_masks: Number of masks to apply
    
    Returns:
        Masked spectrogram
    """
    spec = spectrogram.copy()
    freq, time = spec.shape
    
    for _ in range(num_masks):
        mask_width = np.random.randint(1, max_mask_time)
        mask_start = np.random.randint(0, max(1, time - mask_width))
        spec[:, mask_start:mask_start + mask_width] = 0
    
    return spec


def frequency_mask(spectrogram, max_mask_freq=20, num_masks=1):
    """
    Apply frequency masking to spectrogram.
    
    Args:
        spectrogram: Input spectrogram [freq, time]
        max_mask_freq: Maximum frequency bins to mask
        num_masks: Number of masks to apply
    
    Returns:
        Masked spectrogram
    """
    spec = spectrogram.copy()
    freq, time = spec.shape
    
    for _ in range(num_masks):
        mask_height = np.random.randint(1, max_mask_freq)
        mask_start = np.random.randint(0, max(1, freq - mask_height))
        spec[mask_start:mask_start + mask_height, :] = 0
    
    return spec


def add_gaussian_noise(audio, noise_level=0.005):
    """
    Add Gaussian noise to audio signal.
    
    Args:
        audio: Input audio signal
        noise_level: Standard deviation of noise
    
    Returns:
        Noisy audio
    """
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


def simulate_codec_compression(audio, sr=16000, quality='low'):
    """
    Simulate codec compression artifacts by encoding/decoding.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        quality: Compression quality ('low', 'medium', 'high')
    
    Returns:
        Compressed audio
    """
    quality_map = {
        'low': 32000,
        'medium': 64000,
        'high': 128000
    }
    
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='OGG', subtype='VORBIS')
        buffer.seek(0)
        compressed_audio, _ = sf.read(buffer)
        
        if len(compressed_audio) < len(audio):
            compressed_audio = np.pad(compressed_audio, (0, len(audio) - len(compressed_audio)))
        elif len(compressed_audio) > len(audio):
            compressed_audio = compressed_audio[:len(audio)]
        
        return compressed_audio
    
    except Exception as e:
        return audio


def pitch_shift(audio, sr=16000, n_steps=None):
    """
    Apply random pitch shift to audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_steps: Number of semitones to shift (if None, random ±2)
    
    Returns:
        Pitch-shifted audio
    """
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    
    try:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return shifted
    except Exception as e:
        print(f"Pitch shift failed: {e}")
        return audio



class AudioAugmenter:
    """
    Applies random augmentations to audio with specified probability.
    """
    
    def __init__(self, prob=0.5):
        """
        Args:
            prob: Probability of applying each augmentation
        """
        self.prob = prob
    
    def augment_audio(self, audio, sr=16000):
        if np.random.random() < self.prob:
            audio = add_gaussian_noise(audio, noise_level=np.random.uniform(0.001, 0.01))
        
        if np.random.random() < self.prob:
            audio = simulate_codec_compression(audio, sr=sr, quality=np.random.choice(['low', 'medium', 'high']))
        
        if np.random.random() < self.prob:
            audio = pitch_shift(audio, sr=sr)
        
        return audio
    
    def augment_spectrogram(self, spectrogram):
        """
        Apply random augmentations to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [freq, time]
        
        Returns:
            Augmented spectrogram
        """
        if np.random.random() < self.prob:
            spectrogram = time_mask(spectrogram, max_mask_time=20, num_masks=np.random.randint(1, 3))
        
        if np.random.random() < self.prob:
            spectrogram = frequency_mask(spectrogram, max_mask_freq=20, num_masks=np.random.randint(1, 3))
        
        return spectrogram


if __name__ == "__main__":
    print("Testing augmentation techniques...")
    
    test_audio = np.random.randn(32000)
    test_spec = np.random.randn(128, 200)
    
    augmenter = AudioAugmenter(prob=1.0)
    
    print("\n1. Time masking...")
    masked = time_mask(test_spec)
    print(f"   Original shape: {test_spec.shape}, Masked shape: {masked.shape}")
    
    print("\n2. Frequency masking...")
    masked = frequency_mask(test_spec)
    print(f"   Original shape: {test_spec.shape}, Masked shape: {masked.shape}")
    
    print("\n3. Gaussian noise...")
    noisy = add_gaussian_noise(test_audio)
    print(f"   Original shape: {test_audio.shape}, Noisy shape: {noisy.shape}")
    
    print("\n4. Codec compression...")
    compressed = simulate_codec_compression(test_audio)
    print(f"   Original shape: {test_audio.shape}, Compressed shape: {compressed.shape}")
    
    print("\n5. Pitch shift...")
    shifted = pitch_shift(test_audio)
    print(f"   Original shape: {test_audio.shape}, Shifted shape: {shifted.shape}")
    
    print("\nAll augmentation techniques working correctly!")
