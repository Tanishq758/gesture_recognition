"""
landmarks.py — Hand landmark normalization.

IMPORTANT: This function must be identical to the one used in
collect_data.py during training. Any change here breaks the model.
"""


def extract_landmarks(hand_landmarks) -> list:
    """
    Normalize 21 MediaPipe hand landmarks relative to the wrist (landmark 0),
    scaled by the distance from wrist to middle finger MCP (landmark 9).

    Returns a flat list of 63 floats: [x0, y0, z0, x1, y1, z1, ...]
    Values are scale-invariant and position-invariant.
    """
    landmarks = hand_landmarks.landmark
    wrist     = landmarks[0]
    wx, wy, wz = wrist.x, wrist.y, wrist.z

    ref   = landmarks[9]
    scale = ((ref.x - wx)**2 + (ref.y - wy)**2 + (ref.z - wz)**2) ** 0.5
    if scale == 0:
        scale = 1

    row = []
    for lm in landmarks:
        row.append((lm.x - wx) / scale)
        row.append((lm.y - wy) / scale)
        row.append((lm.z - wz) / scale)
    return row