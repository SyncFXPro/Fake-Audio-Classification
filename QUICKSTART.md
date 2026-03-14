# Quick Start Guide

Get your deepfake audio detection system up and running in minutes.

## Prerequisites

- Python 3.8+
- Node.js 16+
- NVIDIA GPU with CUDA support (for training)
- 32GB VRAM (dual RTX 5060 Ti recommended)

## Step 1: Install Dependencies

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Step 2: Train the Model

```bash
cd backend
python training.py
```

**What happens:**
- Loads 13,956 training samples
- Trains on dual GPUs with DataParallel
- Applies 5 augmentation techniques
- Saves best model to `checkpoints/best_model.pth`
- Generates 5 validation graphs in `results/`

**Time:** ~2-3 hours on dual RTX 5060 Ti

**Expected output:**
```
CUDA available: True
GPU count: 2
GPU 0: NVIDIA GeForce RTX 5060 Ti
GPU 1: NVIDIA GeForce RTX 5060 Ti

Loading datasets...
Loaded 13956 files for train split
  - Fake: 6978 samples
  - Real: 6978 samples

Using 2 GPUs with DataParallel
Starting training for 40 epochs...

Epoch 1/40
Training: 100%|████████| 218/218 [02:15<00:00]
Train Loss: 0.4523, Train Acc: 0.7834
Val Loss: 0.3421, Val Acc: 0.8567, Val F1: 0.8512

...

Best F1: 0.9347
Saved checkpoint with F1: 0.9347

=== Validation Test Results ===
Total samples: 1088
Accuracy: 93.47%
Precision (Fake): 0.94
Recall (Fake): 0.92
F1 Score (Fake): 0.93
AUC-ROC: 0.97

Graphs saved to backend/results/
```

## Step 3: Start the Backend

```bash
cd backend
python server.py
```

**Output:**
```
==================================================
Deepfake Audio Detection API Server
==================================================
Loading model on device: cuda
Model loaded successfully! F1 Score: 0.9347

Starting Flask server on http://localhost:5000
API Endpoints:
  - GET  /api/model/status
  - POST /api/predict/upload
  - POST /api/predict/realtime
  - GET  /api/health
==================================================
```

## Step 4: Start the Frontend

In a **new terminal**:

```bash
cd frontend
npm run dev
```

**Output:**
```
VITE v8.0.0  ready in 153 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

## Step 5: Use the Application

1. Open `http://localhost:5173` in your browser

2. **Upload Mode:**
   - Drag & drop an audio file
   - Or click "Browse Files"
   - Supported: MP3, WAV, OGG
   - Get instant results

3. **Record Mode:**
   - Click "Record Audio" tab
   - Click "Start Recording"
   - Speak for 2+ seconds
   - Click "Stop Recording"
   - Get instant results

## Understanding Results

### Fakeness Score
- **0-30%**: Likely REAL audio
- **30-70%**: Uncertain (review manually)
- **70-100%**: Likely FAKE audio

### Prediction
- **REAL**: Model is confident the audio is authentic
- **FAKE**: Model detected deepfake characteristics

### Confidence
- How certain the model is about its prediction
- Higher confidence = more reliable result

## Validation Graphs

After training, check `backend/results/` for:

1. **training_history.png** - Loss and accuracy over time
2. **roc_curve.png** - ROC curve with AUC score
3. **confusion_matrix.png** - Classification performance
4. **probability_distribution.png** - Model confidence distribution
5. **precision_recall_curve.png** - Precision vs recall

## Testing the API

### Check Model Status
```bash
curl http://localhost:5000/api/model/status
```

### Upload File
```bash
curl -X POST -F "file=@audio.mp3" http://localhost:5000/api/predict/upload
```

### Response Format
```json
{
  "fakeness_score": 0.87,
  "prediction": "fake",
  "confidence": 0.87
}
```

## Common Issues

### "Model not loaded"
**Solution:** Train the model first with `python training.py`

### "CUDA out of memory"
**Solution:** Reduce batch size in `training.py`:
```python
train_loader, val_loader, test_loader = get_dataloaders(
    root_dir, batch_size=32, num_workers=4  # Reduced from 64, 8
)
```

### Frontend can't connect
**Solution:** 
1. Check Flask is running on port 5000
2. Check CORS is enabled in `server.py`
3. Verify `API_BASE_URL` in `main.js`

### Microphone not working
**Solution:**
1. Grant microphone permissions in browser
2. Use HTTPS or localhost
3. Check browser console for errors

## Next Steps

1. **Fine-tune the model** - Adjust hyperparameters in `training.py`
2. **Add more data** - Expand the dataset for better accuracy
3. **Deploy to production** - Use Gunicorn/Nginx for Flask, build frontend
4. **Monitor performance** - Track predictions and retrain periodically

## Support

For issues or questions:
1. Check the full README.md
2. Review the plan file
3. Check training logs in `results/`
4. Verify dataset structure

## Performance Tips

### Training
- Use dual GPUs for 2x speedup
- Increase batch size if you have more VRAM
- Enable mixed precision training for faster training

### Inference
- Use GPU for faster predictions
- Batch multiple files together
- Cache model in memory (already done)

### Frontend
- Enable gzip compression
- Use CDN for static assets
- Implement client-side caching

---

**Congratulations!** Your deepfake audio detection system is now running.
