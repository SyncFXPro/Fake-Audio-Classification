import torch
import torch.nn as nn
from model import SpectrogramCNN
from dataset import get_dataloaders
from training import run_validation_test

print("Loading model and dataset...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_dir = r"H:\FAC\Dataset"

_, _, test_loader = get_dataloaders(root_dir, batch_size=64, num_workers=0)

model = SpectrogramCNN()
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

print(f"Model loaded with F1 score: {checkpoint.get('f1_score', 'N/A')}")

run_validation_test(model, test_loader, device=device)

print("\nValidation graphs generated successfully!")
