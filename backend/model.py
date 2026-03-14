import torch
import torch.nn as nn

class SpectrogramCNN(nn.Module):
    """
    CNN for spectrogram-based deepfake audio detection.
    
    Architecture:
    - 4 convolutional blocks with increasing filters (32, 64, 128, 256)
    - Each block: Conv2D -> BatchNorm -> ReLU -> MaxPool
    - Adaptive average pooling to fixed size
    - Classifier: FC(256->128) -> Dropout -> FC(128->1) -> Sigmoid
    
    Input: [batch, 1, 128, 200] (Log-Mel spectrogram)
    Output: [batch, 1] (fakeness score 0-1)
    """
    
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #one!
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #two!
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #three!
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        #four!
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, 1, 128, time]
        
        Returns:
            output: Fakeness score [batch, 1]
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing SpectrogramCNN model...")
    
    model = SpectrogramCNN()
    
    print(f"\nModel architecture:")
    print(model)
    
    total_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Model size: ~{total_params / 1e6:.2f}M parameters")
    
    dummy_input = torch.randn(4, 1, 128, 200)
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\nModel working correctly!")
