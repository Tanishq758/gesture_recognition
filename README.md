# 🤟 Gesture to Language — Real-Time Hand Gesture Recognition System

> Convert hand gestures to natural language sentences in real time using deep learning, MediaPipe, and LLM post-processing.

**Live Demo → [huggingface.co/spaces/Tanishq14/gesture-recognition](https://huggingface.co/spaces/Tanishq14/gesture-recognition)**

---

## Demo

> *(Record a 60 second GIF and place it here — show gestures → words → SEND → sentence spoken aloud)*

---

## Overview

A complete end-to-end real-time hand gesture recognition system built for accessibility. The user signs gestures in front of a webcam, the system recognizes them as words, and an LLM converts the word sequence into a grammatically correct spoken sentence.

The system runs at real-time speed with zero-lag landmark overlay — MediaPipe runs entirely in the browser, while ONNX Runtime handles model inference on the server without any TensorFlow dependency.

---

## Architecture

```
Browser (MediaPipe JS)
        │
        │  63 normalized landmark floats (not raw frames)
        │
        ▼
FastAPI WebSocket Server
        │
        ├── Rolling 20-frame buffer
        ├── 1D CNN (ONNX Runtime) inference
        ├── Stability filter (3 consecutive same predictions)
        ├── Cooldown logic (1 second between words)
        │
        ▼
Word Buffer → Groq LLaMA 3.1 (with conversation history)
        │
        ▼
Sentence + Voice Output (Web Speech API)
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Gesture detection | MediaPipe Hands (JS) | Runs in browser — zero network lag for landmarks |
| Model format | ONNX Runtime | No TensorFlow at runtime — faster, lighter deployment |
| Backend | FastAPI + WebSocket | Persistent connection handles 15fps landmark stream |
| LLM | Groq LLaMA 3.1 8B | Free tier, ~300ms inference, conversation-aware |
| Voice | Web Speech API | Built into browser — no external TTS service needed |
| Deployment | Hugging Face Spaces | Free forever, no cold starts, ML-native platform |

---

## Model Comparison

Three architectures were trained and benchmarked on the same self-collected dataset:

| Model | Val Accuracy | Inference Time | Parameters | Selected |
|-------|-------------|----------------|------------|----------|
| 1D CNN | 97.9% | ~12ms | ~45K | ✅ |
| LSTM | 98.2% | ~28ms | ~180K | ❌ |
| GRU | 97.4% | ~22ms | ~135K | ❌ |

**Why CNN was selected over LSTM despite lower accuracy:**
LSTM had marginally higher validation accuracy but CNN had 4x fewer parameters and 2.3x faster inference. For a real-time system running at 15fps, inference latency matters more than a sub-1% accuracy difference. CNN also converged in fewer epochs with less overfitting.

---

## Gesture Vocabulary (31 gestures)

| Category | Gestures |
|----------|---------|
| Greetings | HELLO, YES, NO, PLEASE, SORRY, OK, WAIT, THANKS |
| Commands | STOP, START, NEXT, PREVIOUS, SELECT, CANCEL, HELP |
| Pronouns | ME, YOU |
| Activities | COME, GO, WORK, EAT, SLEEP, DRINK |
| Emotions | HAPPY, SAD, GOOD, BAD |
| Needs | WATER, FOOD, HOME, TIME |
| Control | SEND, CLEAR, DELETE |

---

## Key Engineering Decisions

**MediaPipe moved to browser**
Originally MediaPipe ran server-side. Every frame was encoded as JPEG, sent over WebSocket, decoded, and processed. This caused visible lag. Moving MediaPipe to JavaScript means landmarks are drawn instantly with zero network delay. The server now receives 63 pre-normalized floats instead of full JPEG frames — drastically reducing bandwidth and server load.

**ONNX Runtime instead of TensorFlow**
Model was trained in TensorFlow on Google Colab and exported via `tf2onnx`. The server runs inference using ONNX Runtime which has no TensorFlow dependency, installs in seconds, and runs faster on CPU.

**Stability filter for flickering**
Similar gestures (e.g. HELLO and STOP) produced alternating predictions. A stability buffer requires the model to predict the same word 3 consecutive times before confirming it — eliminating false positives without affecting responsiveness.

**Landmark normalization must match training**
All landmarks are normalized relative to the wrist (landmark 0) and scaled by the distance to landmark 9 (middle finger MCP). This makes predictions invariant to hand size and camera distance. The exact same normalization function is used in both training and inference — any deviation breaks the model.

**Frame flip matching**
Training data was collected with `cv2.flip(frame, 1)`. MediaPipe JS gives unflipped coordinates. The JavaScript normalization function mirrors x-coordinates (`x = 1 - x`) before normalizing to match the training coordinate space exactly.

---

## Project Structure

```
gesture_recognition/
├── main.py           — FastAPI routes, WebSocket, Groq endpoint
├── config.py         — all constants (sequence length, thresholds)
├── landmarks.py      — landmark normalization (must match training)
├── model_utils.py    — ONNX model loading and inference
├── session.py        — per-connection state (frame buffer, word buffer)
├── index.html        — frontend (MediaPipe JS, canvas, WebSocket client)
├── Dockerfile        — container config for HF Spaces deployment
├── requirements.txt  — server dependencies only (no TensorFlow)
├── models/
│   ├── cnn_model.onnx
│   └── label_encoder.pkl
└── src/
    ├── collect_data.py   — data collection script
    └── inference.py      — local inference script (development only)
```

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/Tanishq14/gesture-recognition.git
cd gesture-recognition
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

**4. Start the server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**5. Open the app**

Visit `http://localhost:8000` in your browser. Allow webcam access when prompted.

---

## How It Works

1. Browser captures webcam feed and runs MediaPipe Hands locally
2. On detecting the right hand, landmarks are normalized using `extractLandmarks()` — 63 floats
3. Normalized landmarks are sent to the FastAPI server over WebSocket at 15fps
4. Server appends landmarks to a rolling 20-frame buffer (deque)
5. When buffer is full, ONNX Runtime runs inference — returns predicted gesture and confidence
6. Stability filter checks if the same gesture was predicted 3 times consecutively
7. If stable and cooldown has passed (1 second), the word is added to the word buffer
8. User clicks SEND — word buffer is sent to Groq LLaMA with conversation history as context
9. Groq returns a grammatically correct sentence
10. Sentence is displayed and read aloud via Web Speech API

---

## Limitations and Future Work

- Right hand only — left hand support requires mirroring x-coordinates during training
- 31 gesture vocabulary — expanding requires new data collection and retraining
- Transition frames between gestures can occasionally cause mispredictions
- Model trained on single user — performance may vary across different hand sizes and skin tones

---

## Dependencies

```
fastapi          — async web framework
uvicorn          — ASGI server with WebSocket support
onnxruntime      — model inference without TensorFlow
opencv-python-headless — frame processing (server side)
groq             — LLaMA 3.1 API client
python-dotenv    — environment variable management
scikit-learn     — label encoder (saved with sklearn)
```

---

## Author

**Tanishq Gupta**
BTech Student | AI/ML Enthusiast

[GitHub](https://github.com/Tanishq14) · [Hugging Face](https://huggingface.co/Tanishq14)

---

*Built as a complete end-to-end ML engineering project — from data collection and model training to WebSocket backend and cloud deployment.*
