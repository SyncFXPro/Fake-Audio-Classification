# Deepfake Audio Detection - Research Project

**Status:** ✅ Complete | **Training F1:** 99.79% (validation) | **Test F1:** 39.8% | **Demonstrates:** Generalization challenges in deepfake detection

---

## Quick Start

### Backend
```bash
cd backend
python server.py
```

### Frontend
```bash
cd frontend
npm run dev
```

Open **http://localhost:5174** in your browser.

---

## Important Disclaimer

⚠️ This detector is trained on **2020-2022 era TTS systems**. It achieves **99% accuracy** on training-era systems but approximately **60% accuracy** on modern AI voice synthesis (ElevenLabs, Resemble AI, etc.). This demonstrates the fundamental challenge in deepfake detection: models learn system-specific artifacts rather than general "fakeness."

---

## What This Project Is

An **honest, AI-assisted research project** that:
- ✅ Achieves 99.79% F1 on training distribution
- ✅ Demonstrates complete ML pipeline (data → training → deployment)
- ✅ Reveals real-world generalization challenges
- ✅ Provides comprehensive analysis and visualizations
- ✅ Includes honest assessment of limitations

**"I did what I could. This is an AI-assisted project, but it's an honest one because I did the work of learning, structuring, and lecturing myself on its logic afterward."**

---

## Features

- 🎤 **Real-time audio recording** with waveform visualization
- 📁 **File upload** with drag & drop support
- 📊 **Complete training analysis** with 5 visualization graphs
- 🔬 **Technical documentation** with architecture details
- ⚡ **Fast inference** (<100ms per clip on GPU)
- 🎯 **Honest disclaimers** about limitations

---

For complete documentation, see **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**
