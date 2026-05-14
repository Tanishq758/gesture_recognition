import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

# ── MediaPipe setup ──────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Config ───────────────────────────────────────────────────
GESTURE_NAME    = "TIME"  # ← change this for each gesture
SEQUENCES       = 100          # number of sequences to collect
FRAMES          = 20           # frames per sequence
SAVE_PATH       = "data/dynamic_data.csv"

# ── Landmark extraction with wrist normalization ─────────────
def extract_landmarks(hand_landmarks):
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

# ── CSV setup ────────────────────────────────────────────────
# Each row = one frame
# Columns: sequence_id, frame_num, x1,y1,z1...x21,y21,z21, label
header = ["sequence_id", "frame_num"]
for i in range(1, 22):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

file_exists = os.path.exists(SAVE_PATH)

# figure out starting sequence_id so we never overwrite old data
start_seq = 0
if file_exists:
    with open(SAVE_PATH, "r") as f:
        rows = f.readlines()
        if len(rows) > 1:                        # at least one data row
            last_row  = rows[-1].strip().split(",")
            start_seq = int(last_row[0]) + 1     # continue from last sequence

if not file_exists:
    with open(SAVE_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)

# ── Main loop ────────────────────────────────────────────────
cap          = cv2.VideoCapture(0)
seq_count    = 0
state        = "WAITING"        # WAITING → COUNTDOWN → COLLECTING → DONE

print(f"\nGesture : {GESTURE_NAME}")
print(f"Collect : {SEQUENCES} sequences × {FRAMES} frames")
print("Press 'S' to start, 'Q' to quit\n")

current_seq    = start_seq
frame_count    = 0
countdown      = 3
countdown_start = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False
    hand_landmarks_data = None

    # ── Detect right hand only ───────────────────────────────
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness != "Right":
                continue

            hand_detected = True
            hand_landmarks_data = hand_landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # ── State machine ────────────────────────────────────────

    if state == "WAITING":
        msg = "Press S to start"
        cv2.putText(frame, msg, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif state == "COUNTDOWN":
        elapsed = time.time() - countdown_start
        remaining = 1 - int(elapsed)
        if remaining <= 0:
            state       = "COLLECTING"
            frame_count = 0
        else:
            cv2.putText(frame, f"Get Ready: {remaining}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    elif state == "COLLECTING":
        if hand_detected and hand_landmarks_data:
            row = extract_landmarks(hand_landmarks_data)
            full_row = [current_seq, frame_count] + row + [GESTURE_NAME]

            with open(SAVE_PATH, "a", newline="") as f:
                csv.writer(f).writerow(full_row)

            frame_count += 1
            time.sleep(0.05)

            # draw progress bar
            progress = int((frame_count / FRAMES) * 300)
            cv2.rectangle(frame, (10, 80), (310, 105), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 80), (10 + progress, 105), (0, 255, 0), -1)
            cv2.putText(frame, f"Frame {frame_count}/{FRAMES}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if frame_count >= FRAMES:
                seq_count   += 1
                current_seq += 1
                print(f"  Sequence {seq_count}/{SEQUENCES} done")

                if seq_count >= SEQUENCES:
                    state = "DONE"
                else:
                    # short pause between sequences then countdown again
                    time.sleep(0.5)
                    state          = "COUNTDOWN"
                    countdown_start = time.time()
                    frame_count    = 0
        else:
            cv2.putText(frame, "NO HAND DETECTED", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    elif state == "DONE":
        cv2.putText(frame, f"DONE! {SEQUENCES} sequences collected", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ── Always show gesture name and sequence count ───────────
    cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Seq: {seq_count}/{SEQUENCES}", (10, 110) if state != "COLLECTING" else (320, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if state == "DONE":
        break

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and state == "WAITING":
        state          = "COUNTDOWN"
        countdown_start = time.time()
        print("Starting countdown...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Collection complete for '{GESTURE_NAME}'")
print(f"   Sequences collected : {seq_count}")
print(f"   Rows added to CSV   : {seq_count * FRAMES}")