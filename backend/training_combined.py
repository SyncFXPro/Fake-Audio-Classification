import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import SpectrogramCNN
from dataset_combined import get_combined_dataloaders


class Trainer:
    """Trainer class for deepfake audio detection model."""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=3e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.early_stop_patience = 5
        
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (spectrograms, labels) in enumerate(pbar):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(spectrograms)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for spectrograms, labels in tqdm(self.val_loader, desc='Validation'):
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        return epoch_loss, epoch_acc, f1, precision, recall
    
    def save_checkpoint(self, epoch, f1_score):
        """Save model checkpoint."""
        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'history': self.history
        }
        
        torch.save(checkpoint, 'checkpoints/best_model_combined.pth')
        print(f"Saved checkpoint with F1: {f1_score:.4f}")
    
    def train(self, epochs=40):
        """Train the model for specified epochs."""
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.module.count_parameters() if hasattr(self.model, 'module') else self.model.count_parameters():,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1, val_precision, val_recall = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
            
            self.scheduler.step(val_f1)
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_f1)
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.early_stop_patience}")
            
            if self.patience_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        with open('results/training_history_combined.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed! Best F1: {self.best_f1:.4f}")


def run_validation_test(model, test_loader, device='cuda'):
    """
    Run comprehensive validation test with metrics and visualizations.
    """
    print("\n" + "="*50)
    print("Running Validation Test on Combined Dataset")
    print("="*50)
    
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(test_loader, desc='Testing'):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            outputs = model(spectrograms)
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
    
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)
    
    print("\n=== Validation Test Results ===")
    print(f"Total samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision (Fake): {precision:.4f}")
    print(f"Recall (Fake): {recall:.4f}")
    print(f"F1 Score (Fake): {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    metrics = {
        'total_samples': len(all_labels),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'confusion_matrix': cm.tolist()
    }
    
    with open('results/validation_metrics_combined.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    generate_validation_plots(all_labels, all_probs, all_preds, cm, suffix='_combined')
    
    print("\nGraphs saved to backend/results/")


def generate_validation_plots(labels, probs, preds, cm, suffix=''):
    """Generate all validation plots."""
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    if os.path.exists(f'results/training_history{suffix}.json'):
        with open(f'results/training_history{suffix}.json', 'r') as f:
            history = json.load(f)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss (Combined Dataset)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy (Combined Dataset)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/training_history{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Combined Dataset)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/roc_curve{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Combined Dataset)')
    plt.savefig(f'results/confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]
    plt.hist(real_probs, bins=50, alpha=0.6, color='green', label='Real Audio', edgecolor='black')
    plt.hist(fake_probs, bins=50, alpha=0.6, color='red', label='Fake Audio', edgecolor='black')
    plt.xlabel('Predicted Fakeness Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution (Combined Dataset)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/probability_distribution{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, 'b-', linewidth=2, 
             label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Combined Dataset)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/precision_recall_curve{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Training on COMBINED Dataset")
    print("(for-original + for-norm + for-rerec)")
    print("="*60)
    
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    root_dir = r"H:\FAC\Dataset"
    
    print("\nLoading combined datasets...")
    train_loader, val_loader, test_loader = get_combined_dataloaders(
        root_dir, batch_size=64, num_workers=8
    )
    
    print("\nInitializing model...")
    model = SpectrogramCNN()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    trainer = Trainer(model, train_loader, val_loader, device=device, lr=3e-4)
    
    print("\n" + "="*60)
    print("Expected training time: 8-10 hours with ~132K samples")
    print("="*60)
    
    trainer.train(epochs=40)
    
    print("\nLoading best model for validation test...")
    checkpoint = torch.load('checkpoints/best_model_combined.pth')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    run_validation_test(model, test_loader, device=device)
    
    print("\n" + "="*60)
    print("Training and validation complete!")
    print("Model saved to: checkpoints/best_model_combined.pth")
    print("Graphs saved to: results/*_combined.png")
    print("="*60)
