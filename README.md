# Deepfake Audio Detection System

AI-powered audio authenticity verification using CNN-based spectrogram analysis.

## Overview

This system uses a Convolutional Neural Network (CNN) to detect deepfake audio by analyzing Log-Mel spectrograms. The model is trained on 2-second audio clips and can distinguish between real and AI-generated audio with high accuracy.

## Architecture

### Model
- **Input**: Log-Mel spectrogram (1x128x200)
- **Architecture**: 4 convolutional blocks with increasing filters (32, 64, 128, 256)
- **Output**: Fakeness score (0-1)
- **Parameters**: ~1.5M trainable parameters
- **Model Size**: ~6MB

### Dataset
- **Training**: 13,956 samples (6,978 fake + 6,978 real)
- **Testing**: 1,088 samples (544 fake + 544 real)
- **Format**: 2-second WAV files, preprocessed at 16kHz

### Performance
- **Expected Accuracy**: 90-95%
- **Expected F1 Score**: 0.88-0.93
- **Inference Time**: <100ms per clip
- **Training Time**: 2-3 hours on dual RTX 5060 Ti

## Project Structure

```
FAC/
├── backend/
│   ├── model.py              # CNN architecture
│   ├── preprocessing.py      # Spectrogram generation
│   ├── augmentation.py       # Data augmentation (5 techniques)
│   ├── dataset.py           # PyTorch Dataset & DataLoader
│   ├── training.py          # Training loop with validation
│   ├── server.py            # Flask API
│   ├── main.py              # Main orchestrator
│   ├── requirements.txt     # Python dependencies
│   ├── checkpoints/         # Saved models
│   └── results/             # Training graphs & metrics
├── frontend/
│   ├── index.html           # UI structure
│   ├── style.css            # Styling
│   ├── main.js              # Frontend logic
│   └── package.json         # Node dependencies
└── Dataset/
    └── for-2sec/for-2seconds/
        ├── training/        # Training data
        ├── testing/         # Test data
        └── validation/      # Validation data
```

## Installation

### Backend Setup

1. Create a virtual environment:
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

## Usage

### 1. Train the Model

Train the model on your dataset:

```bash
cd backend
python training.py
```

This will:
- Load the dataset from `H:\FAC\Dataset`
- Train using dual GPUs (RTX 5060 Ti) with DataParallel
- Apply 5 augmentation techniques during training
- Save the best model to `checkpoints/best_model.pth`
- Generate validation graphs in `results/`

Training takes approximately 2-3 hours.

### 2. Start the Backend Server

```bash
cd backend
python server.py
```

The Flask API will start on `http://localhost:5000`

### 3. Start the Frontend

In a separate terminal:

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:5173`

### 4. Use the Application

Open `http://localhost:5173` in your browser.

**Upload Mode:**
- Drag & drop an audio file or click to browse
- Supported formats: MP3, WAV, OGG
- Get instant analysis results

**Record Mode:**
- Click "Start Recording"
- Record at least 2 seconds of audio
- Click "Stop Recording"
- Get instant analysis results

## API Endpoints

### GET `/api/model/status`
Check if model is loaded and get accuracy metrics.

**Response:**
```json
{
  "loaded": true,
  "device": "cuda",
  "accuracy": 0.93,
  "version": "1.0"
}
```

### POST `/api/predict/upload`
Upload an audio file for analysis.

**Request:** `multipart/form-data` with file

**Response:**
```json
{
  "fakeness_score": 0.87,
  "prediction": "fake",
  "confidence": 0.87
}
```

### POST `/api/predict/realtime`
Analyze recorded audio from microphone.

**Request:**
```json
{
  "audio": "base64_encoded_audio_data"
}
```

**Response:**
```json
{
  "fakeness_score": 0.23,
  "prediction": "real",
  "confidence": 0.77
}
```

## Features

### Data Augmentation
1. **Time Masking** - Masks random time frames
2. **Frequency Masking** - Masks random mel bins
3. **Gaussian Noise** - Adds random noise
4. **Codec Compression** - Simulates MP3/AAC artifacts
5. **Pitch Shift** - Random pitch changes (±2 semitones)

### Multi-GPU Training
- Automatic detection of available GPUs
- DataParallel for batch splitting across GPUs
- Optimized batch size (64 total, 32 per GPU)
- ~2x training speedup with dual GPUs

### Validation Testing
After training, comprehensive validation generates:
1. **Training History** - Loss and accuracy curves
2. **ROC Curve** - With AUC score
3. **Confusion Matrix** - True/False positives/negatives
4. **Probability Distribution** - Model confidence visualization
5. **Precision-Recall Curve** - Performance across thresholds

All graphs saved to `backend/results/`

## Technical Details

### Preprocessing
- Sample rate: 16kHz
- FFT size: 1024
- Hop length: 256
- Mel bins: 128
- Normalization: Per-sample z-score

### Training Configuration
- Loss: BCELoss
- Optimizer: AdamW (lr=3e-4)
- Scheduler: ReduceLROnPlateau (patience=3)
- Batch size: 64 (32 per GPU)
- Epochs: 30-40
- Early stopping: patience=5
- Checkpointing: Best F1 score

### Model Architecture
```
Input [1, 128, 200]
    ↓
Conv Block 1: 1→32 filters
    ↓
Conv Block 2: 32→64 filters
    ↓
Conv Block 3: 64→128 filters
    ↓
Conv Block 4: 128→256 filters
    ↓
AdaptiveAvgPool (1x1)
    ↓
FC: 256→128 + Dropout(0.3)
    ↓
FC: 128→1 + Sigmoid
    ↓
Output: Fakeness score [0-1]
```

## Troubleshooting

### Model not loading
- Ensure you've trained the model first: `python training.py`
- Check that `checkpoints/best_model.pth` exists

### CUDA out of memory
- Reduce batch size in `training.py` (default: 64)
- Reduce num_workers in DataLoader (default: 8)

### Frontend can't connect to backend
- Ensure Flask server is running on port 5000
- Check CORS settings in `server.py`
- Verify API_BASE_URL in `main.js`

### Audio recording not working
- Grant microphone permissions in browser
- Use HTTPS or localhost (required for getUserMedia)

## License

See LICENSE file for details.

## Acknowledgments

- Dataset: Fake-or-Real Audio Dataset
- Framework: PyTorch
- Frontend: Vanilla JavaScript with Vite
- Backend: Flask
