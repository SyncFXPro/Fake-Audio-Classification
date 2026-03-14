import os
import sys
import subprocess
import time

def check_model_exists():
    """Check if trained model exists."""
    checkpoint_path = 'checkpoints/best_model.pth'
    return os.path.exists(checkpoint_path)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import flask
        import librosa
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def main():
    """Main orchestrator for the deepfake audio detection system."""
    print("="*60)
    print("Deepfake Audio Detection System")
    print("="*60)
    
    if not check_dependencies():
        print("\nError: Missing dependencies!")
        print("Install them with: pip install -r requirements.txt")
        return
    
    if not check_model_exists():
        print("\nWarning: Trained model not found!")
        print("\nYou need to train the model first.")
        print("Run: python training.py")
        print("\nThis will:")
        print("  1. Load the dataset from H:\\FAC\\Dataset")
        print("  2. Train the CNN model on dual GPUs")
        print("  3. Save the best model to checkpoints/")
        print("  4. Generate validation graphs in results/")
        print("\nTraining will take approximately 2-3 hours on dual RTX 5060 Ti.")
        
        response = input("\nDo you want to start training now? (y/n): ")
        if response.lower() == 'y':
            print("\nStarting training...")
            subprocess.run([sys.executable, 'training.py'])
        else:
            print("\nExiting. Train the model when ready.")
            return
    
    print("\nModel found! Starting servers...")
    print("\n" + "="*60)
    print("Starting Flask API Server...")
    print("="*60)
    
    print("\nBackend will start on: http://localhost:5000")
    print("Frontend will start on: http://localhost:5173")
    print("\nTo start the system:")
    print("  1. Run this script to start the Flask backend")
    print("  2. In another terminal, run: cd ../frontend && npm run dev")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from server import main as server_main
    server_main()

if __name__ == "__main__":
    main()

