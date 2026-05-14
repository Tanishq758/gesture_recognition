"""
config.py — All constants in one place.
To tune the system, only this file needs to change.
"""

# ── Model paths ───────────────────────────────────────────────
MODEL_PATH         = "models/cnn_model.onnx"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# ── Gesture pipeline ──────────────────────────────────────────
SEQUENCE_LENGTH      = 20     # frames per gesture sequence
CONFIDENCE_THRESHOLD = 0.70   # minimum confidence to consider a prediction
COOLDOWN_SECONDS     = 1.0    # minimum seconds between confirmed words
IGNORED_GESTURES     = ["START"]

# ── Stability filter ──────────────────────────────────────────
# Model must predict the SAME word this many times in a row
# before it gets added to the word buffer.
# Prevents flickering between similar gestures (e.g. HELLO vs STOP).
STABILITY_REQUIRED   = 3