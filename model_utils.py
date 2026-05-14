"""
model_utils.py — ONNX model loading and inference.

Model and label encoder are loaded ONCE at import time and shared
across all WebSocket sessions. ONNX Runtime is read-only during
inference so sharing across sessions is safe.

Named model_utils.py (not inference.py) to avoid clashing with
src/inference.py which is the standalone local inference script.
"""

import pickle

import numpy as np
import onnxruntime as ort

from config import MODEL_PATH, LABEL_ENCODER_PATH

# ── Load once at startup ──────────────────────────────────────
ort_session   = ort.InferenceSession(MODEL_PATH)
INPUT_NAME    = ort_session.get_inputs()[0].name    # e.g. "input_1"
OUTPUT_NAME   = ort_session.get_outputs()[0].name   # e.g. "output_0"

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


# ── Inference ─────────────────────────────────────────────────
def run_inference(frame_buffer: list) -> tuple[str, float]:
    """
    Run ONNX inference on a full 20-frame buffer.

    Args:
        frame_buffer: list of 20 rows, each row is 63 floats
                      (output of extract_landmarks)

    Returns:
        (label, confidence) — top predicted gesture and its probability
    """
    # shape: (1, 20, 63) — batch=1, timesteps=20, features=63
    x       = np.array(frame_buffer, dtype=np.float32)[np.newaxis, ...]
    outputs = ort_session.run([OUTPUT_NAME], {INPUT_NAME: x})
    probs   = outputs[0][0]           # shape: (num_classes,)

    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    label = label_encoder.inverse_transform([idx])[0]

    return label, conf