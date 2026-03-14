import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from preprocessing import audio_to_spectrogram
from augmentation import AudioAugmenter

class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio deepfake detection.
    Loads audio files, converts to spectrograms, and applies augmentations.
    """
    
    def __init__(self, root_dir, split='train', augment=True):
        """
        Args:
            root_dir: Root directory containing train/test/validation folders
            split: 'train', 'test', or 'validation'
            augment: Whether to apply augmentations (only for training)
        """
        self.root_dir = root_dir
        self.split = split
        self.augment = augment and (split == 'train')
        self.augmenter = AudioAugmenter(prob=0.5) if self.augment else None
        
        self.file_paths = []
        self.labels = []
        
        self._load_file_paths()
    
    def _load_file_paths(self):
        """Load all file paths and labels from the dataset directory."""
        split_dir = os.path.join(self.root_dir, 'for-2sec', 'for-2seconds', 
                                  'training' if self.split == 'train' else self.split)
        
        fake_dir = os.path.join(split_dir, 'fake')
        real_dir = os.path.join(split_dir, 'real')
        
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                         if f.endswith('.wav')]
            self.file_paths.extend(fake_files)
            self.labels.extend([1] * len(fake_files))
        
        if os.path.exists(real_dir):
            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                         if f.endswith('.wav')]
            self.file_paths.extend(real_files)
            self.labels.extend([0] * len(real_files))
        
        print(f"Loaded {len(self.file_paths)} files for {self.split} split")
        print(f"  - Fake: {sum(self.labels)} samples")
        print(f"  - Real: {len(self.labels) - sum(self.labels)} samples")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            spectrogram: Tensor of shape [1, 128, time]
            label: 0 for real, 1 for fake
        """
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            if self.augment:
                y, sr = librosa.load(audio_path, sr=16000, duration=2.0)
                y = self.augmenter.augment_audio(y, sr=sr)
                
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128
                )
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                spectrogram = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
                
                spectrogram = self.augmenter.augment_spectrogram(spectrogram)
            else:
                spectrogram = audio_to_spectrogram(audio_path)
            
            spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0)
            label = torch.FloatTensor([label])
            
            return spectrogram, label
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, 128, 200), torch.FloatTensor([label])


def get_dataloaders(root_dir, batch_size=64, num_workers=8):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir: Root directory of dataset
        batch_size: Batch size (will be split across GPUs with DataParallel)
        num_workers: Number of worker processes
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = AudioDataset(root_dir, split='train', augment=True)
    val_dataset = AudioDataset(root_dir, split='validation', augment=False)
    test_dataset = AudioDataset(root_dir, split='testing', augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root_dir = r"H:\FAC\Dataset"
    
    print("Testing AudioDataset...")
    
    try:
        train_dataset = AudioDataset(root_dir, split='train', augment=True)
        print(f"\nTrain dataset size: {len(train_dataset)}")
        
        spec, label = train_dataset[0]
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Label: {label.item()} ({'fake' if label.item() == 1 else 'real'})")
        
        print("\nTesting DataLoader...")
        train_loader, val_loader, test_loader = get_dataloaders(root_dir, batch_size=32, num_workers=0)
        
        batch_spec, batch_labels = next(iter(train_loader))
        print(f"Batch spectrogram shape: {batch_spec.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        
        print("\nDataset module working correctly!")
    
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
