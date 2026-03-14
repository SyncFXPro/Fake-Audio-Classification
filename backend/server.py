import os
import io
import base64
import tempfile
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import soundfile as sf

from model import SpectrogramCNN
from preprocessing import audio_to_spectrogram

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174"])

model = None
device = None

def load_model():
    """Load the trained model from checkpoint."""
    global model, device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on device: {device}")
    
    model = SpectrogramCNN()
    
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully! F1 Score: {checkpoint.get('f1_score', 'N/A')}")
        return True
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first by running: python training.py")
        return False


def predict_audio(audio_data, sr=16000):
    """
    Run inference on audio data.
    
    Args:
        audio_data: Audio signal as numpy array
        sr: Sample rate
    
    Returns:
        dict with fakeness_score, prediction, confidence
    """
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_fft=1024, hop_length=256, n_mels=128
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        spectrogram = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(spec_tensor)
            fakeness_score = output.item()
            
        
        prediction = "fake" if fakeness_score > 0.5 else "real"
        confidence = fakeness_score if fakeness_score > 0.5 else (1 - fakeness_score)
        
        return {
            "fakeness_score": float(fakeness_score),
            "prediction": prediction,
            "confidence": float(confidence)
        }
    
    except Exception as e:
        return {"error": str(e)}


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model status."""
    if model is None:
        return jsonify({
            "loaded": False,
            "message": "Model not loaded. Please train the model first."
        }), 503
    
    checkpoint_path = 'checkpoints/best_model.pth'
    accuracy = "N/A"
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        accuracy = checkpoint.get('f1_score', 'N/A')
    
    return jsonify({
        "loaded": True,
        "device": str(device),
        "accuracy": accuracy,
        "version": "1.0"
    })


@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    """Handle audio file upload and prediction."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        audio_bytes = file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        audio_data, sr = librosa.load(audio_buffer, sr=16000, duration=2.0)
        
        result = predict_audio(audio_data, sr)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500




@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def get_graph_file(name):
    for graph_name in os.listdir(RESULTS_DIR):
        if not graph_name.endswith('.png'):
            continue
        if not graph_name.startswith(name):
            continue
        graph_path = os.path.join(RESULTS_DIR, graph_name)
        if not os.path.exists(graph_path):
            print(f"Graph {graph_name} not found at {graph_path}")
            print("Please generate the graphs first by running: python generate_graphs.py")
            exit(1)
        return graph_path


@app.route('/api/graph/<graph_name>', methods=['GET'])
def serve_graph(graph_name):
    """Serve training/validation graphs from backend/results/."""
    from flask import send_file

    graph_path = get_graph_file(graph_name)

    if not os.path.exists(graph_path):
        return jsonify({"error": "Graph file not found"}), 404

    return send_file(graph_path, mimetype='image/png')





def main():
    """Start the Flask server."""
    print("="*50)
    print("Deepfake Audio Detection API Server")
    print("="*50)
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\nWARNING: Model not loaded!")
        print("Train the model first: python training.py")
        print("\nServer will start but predictions will fail.")
    
    print("\nStarting Flask server on http://localhost:5000")
    print("API Endpoints:")
    print("  - GET  /api/model/status")
    print("  - POST /api/predict/upload")
    print("  - POST /api/predict/realtime")
    print("  - GET  /api/health")
    print("\nMake sure to run the frontend: cd ../frontend && npm run dev")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
   
