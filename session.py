"""
session.py — Per-WebSocket-connection state.

Each browser tab that connects gets its own GestureSession.
MediaPipe now runs in the browser (JavaScript), so this class
only manages inference state — no MediaPipe instance needed.
"""

from collections import deque

from config import SEQUENCE_LENGTH


class GestureSession:
    """All mutable state that belongs to one WebSocket connection."""

    def __init__(self):
        # ── Gesture pipeline state ────────────────────────────
        self.frame_buffer        = deque(maxlen=SEQUENCE_LENGTH)
        self.word_buffer         = []      # confirmed words
        self.last_pred_time      = 0.0     # epoch seconds — for cooldown
        self.last_predicted_word = None    # prevents consecutive duplicates

        # ── Stability filter state ────────────────────────────
        self.stability_buffer    = []      # rolling window of raw predictions
        self.last_confirmed_word = None    # last confirmed word (shown on screen)

    def close(self):
        """Called when WebSocket disconnects. Nothing to clean up now."""
        pass