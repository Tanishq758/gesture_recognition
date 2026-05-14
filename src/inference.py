import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
import onnxruntime as ort

# ── Load model and encoder ────────────────────────────────────
session = ort.InferenceSession('models/cnn_model.onnx')
input_name = session.get_inputs()[0].name

with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# ── MediaPipe setup ───────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Config ────────────────────────────────────────────────────
SEQUENCE_LENGTH      = 20
CONFIDENCE_THRESHOLD = 0.70
COOLDOWN_SECONDS     = 1.0

# ── State ─────────────────────────────────────────────────────
frame_buffer   = []
word_buffer    = []
last_pred_time = 0
current_word   = ""

# ── Normalization (exact copy from collect_data.py) ───────────
def extract_landmarks(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]
    wx, wy, wz = wrist.x, wrist.y, wrist.z
    ref = landmarks[9]
    scale = ((ref.x - wx)**2 + (ref.y - wy)**2 + (ref.z - wz)**2) ** 0.5
    if scale == 0:
        scale = 1
    row = []
    for lm in landmarks:
        row.append((lm.x - wx) / scale)
        row.append((lm.y - wy) / scale)
        row.append((lm.z - wz) / scale)
    return row

# ── Prediction helper ─────────────────────────────────────────
# def predict(buffer):
#     sequence = np.array(buffer, dtype=np.float32)
#     sequence = np.expand_dims(sequence, axis=0)        # (1, 20, 63)
#     output   = session.run(None, {input_name: sequence})
#     probs    = output[0][0]                            # (31,)
#     return probs
def predict(buffer):
    sequence = np.array(buffer, dtype=np.float32)
    sequence = np.expand_dims(sequence, axis=0)  # (1, 20, 63)
    output = session.run(None, {input_name: sequence})
    probs = output[0][0]  # (31,)

    # DEBUG — top 3 predictions
    top3_indices = probs.argsort()[-3:][::-1]
    for i in top3_indices:
        print(f"{le.classes_[i]}: {probs[i]:.2f}")
    print("---")

    return probs
# ── Main loop ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Running. Q=quit | C=clear buffer | D=delete last word")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness != "Right":
                continue

            hand_detected = True
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = extract_landmarks(hand_landmarks)
            frame_buffer.append(row)

            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            current_time  = time.time()
            cooldown_over = (current_time - last_pred_time) >= COOLDOWN_SECONDS

            if len(frame_buffer) == SEQUENCE_LENGTH and cooldown_over:
                probs      = predict(frame_buffer)
                confidence = probs.max()
                pred_index = probs.argmax()

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_word = le.inverse_transform([pred_index])[0]
                    word_buffer.append(predicted_word)
                    current_word   = predicted_word
                    last_pred_time = current_time
                    print(f"Detected: {predicted_word} ({confidence:.2f})")

    if not hand_detected:
        frame_buffer = []

    # ── Display ───────────────────────────────────────────────
    cv2.putText(frame, f"Word: {current_word}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    if hand_detected and len(frame_buffer) == SEQUENCE_LENGTH:
        probs = predict(frame_buffer)
        conf  = probs.max()
        bar   = int(conf * 200)
        cv2.rectangle(frame, (10, 60), (210, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 60), (10 + bar, 80), (0, 255, 0), -1)
        cv2.putText(frame, f"{conf:.2f}", (215, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    buffer_text = " ".join(word_buffer) if word_buffer else "---"
    cv2.putText(frame, f"Buffer: {buffer_text}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, "Q: quit | C: clear | D: delete",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Gesture Inference", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word_buffer  = []
        current_word = ""
        print("Buffer cleared")
    elif key == ord('d') and word_buffer:
        removed      = word_buffer.pop()
        current_word = word_buffer[-1] if word_buffer else ""
        print(f"Deleted: {removed}")

cap.release()
cv2.destroyAllWindows()