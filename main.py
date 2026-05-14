"""
main.py — FastAPI WebSocket backend for Gesture-to-Language System
=================================================================
Run with:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Modular structure:
    config.py      — all constants
    landmarks.py   — normalization function
    model_utils.py — ONNX model + inference
    session.py     — per-connection state (GestureSession)
    main.py        — FastAPI routes only (this file)

Before running, make sure .env contains:
    GROQ_API_KEY=your_key_here
"""

import datetime
import json
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from groq import Groq

from config import (
    CONFIDENCE_THRESHOLD,
    COOLDOWN_SECONDS,
    IGNORED_GESTURES,
    SEQUENCE_LENGTH,
    STABILITY_REQUIRED,
)
from model_utils import label_encoder, run_inference
from session import GestureSession

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

# ── Groq client setup ─────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ── Sentence history ──────────────────────────────────────────────────────────
# Stored in memory — resets when server restarts
# Each entry: { "sentence": str, "words": list, "timestamp": str }
sentence_history: list = []
MAX_HISTORY = 5

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Gesture-to-Language API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Landmark processing ───────────────────────────────────────────────────────
def process_landmarks(session: GestureSession, row: list) -> dict:
    """
    Receives 63 pre-normalized floats from the browser (already extracted
    and normalized by extractLandmarks() in JavaScript).
    Runs ONNX inference when buffer is full.
    No MediaPipe, no image decoding — just pure inference logic.
    """
    # Default response
    response = {
        "hand_detected":  True,   # browser only sends when hand is present
        "prediction":     None,
        "confirmed_word": session.last_confirmed_word,
        "confidence":     0.0,
        "stability":      0,
        "word_buffer":    session.word_buffer,
        "buffer_length":  len(session.frame_buffer),
    }

    # 1. Append normalized row to rolling buffer
    session.frame_buffer.append(row)
    response["buffer_length"] = len(session.frame_buffer)

    # 2. Run inference only when buffer is full
    if len(session.frame_buffer) < SEQUENCE_LENGTH:
        return response

    label, conf = run_inference(list(session.frame_buffer))

    # 3. Stability filter — same word must win STABILITY_REQUIRED times in a row
    if conf >= CONFIDENCE_THRESHOLD and label not in IGNORED_GESTURES:
        session.stability_buffer.append(label)
    else:
        session.stability_buffer = []

    if len(session.stability_buffer) > STABILITY_REQUIRED:
        session.stability_buffer = session.stability_buffer[-STABILITY_REQUIRED:]

    stable = (
        len(session.stability_buffer) == STABILITY_REQUIRED
        and len(set(session.stability_buffer)) == 1
    )

    # 4. Confirm word if stable + cooldown passed
    now         = time.time()
    cooldown_ok = (now - session.last_pred_time) >= COOLDOWN_SECONDS

    if stable and cooldown_ok:
        confirmed_word = session.stability_buffer[0]
        if confirmed_word != session.last_predicted_word:
            session.word_buffer.append(confirmed_word)
            session.last_predicted_word = confirmed_word
        session.last_confirmed_word = confirmed_word
        session.last_pred_time      = now
        session.stability_buffer    = []

    response["prediction"]     = label
    response["confirmed_word"] = session.last_confirmed_word
    response["confidence"]     = round(conf, 4)
    response["stability"]      = len(session.stability_buffer)
    response["word_buffer"]    = session.word_buffer
    return response


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = GestureSession()
    print("[WS] Client connected")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            # Landmark data from browser MediaPipe
            # Browser sends 63 pre-normalized floats — no image processing needed
            if msg.get("type") == "landmarks":
                row    = msg["data"]   # list of 63 floats
                result = process_landmarks(session, row)
                await websocket.send_text(json.dumps(result))

            # No hand in frame — reset all buffers so stale frames don't pollute next gesture
            elif msg.get("type") == "no_hand":
                session.frame_buffer.clear()
                session.stability_buffer    = []
                session.last_confirmed_word = None
                session.last_predicted_word = None
                await websocket.send_text(json.dumps({
                    "hand_detected":  False,
                    "prediction":     None,
                    "confirmed_word": None,
                    "confidence":     0.0,
                    "stability":      0,
                    "buffer_length":  0,
                    "word_buffer":    session.word_buffer,
                }))

            # Legacy frame type — kept for compatibility, ignored now
            elif msg.get("type") == "frame":
                await websocket.send_text(json.dumps({"error": "frame type deprecated, use landmarks"}))

            # Button actions
            elif msg.get("type") == "action":
                action = msg.get("action", "").upper()

                if action == "DELETE":
                    if session.word_buffer:
                        session.word_buffer.pop()

                elif action == "CLEAR":
                    session.word_buffer.clear()
                    session.last_predicted_word = None
                    session.last_confirmed_word = None
                    session.stability_buffer    = []

                await websocket.send_text(json.dumps({
                    "action_ack":  action,
                    "word_buffer": session.word_buffer,
                }))

            else:
                await websocket.send_text(json.dumps({"error": "Unknown message type"}))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        session.close()


# ── Groq LLM endpoint ─────────────────────────────────────────────────────────
@app.post("/gemini")
async def groq_endpoint(payload: dict):
    """
    Receives word buffer → calls Groq LLM → stores in history → returns sentence.
    Endpoint kept as /gemini so frontend needs no changes.

    Request:  { "words": ["HELLO", "MY", "NAME"] }
    Response: { "sentence": "Hello, my name...", "history": [...] }
    """
    words = payload.get("words", [])
    if not words:
        return {"sentence": "", "history": sentence_history}

    # Build messages with conversation history as context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a sign language interpreter assistant. "
                "The user communicates by signing individual words one at a time. "
                "Convert the signed words into a natural, grammatically correct English sentence. "
                "GUIDELINES:\n"
                "1. Keep the core meaning of the signed words intact — do not change the intent.\n"
                "2. You may add small helper words (is, am, are, the, a, to, please, I, you) "
                "to make the sentence flow naturally.\n"
                "3. Light rephrasing is allowed but stay close to what was signed.\n"
                "4. Do not add information that was not implied by the words.\n"
                "5. Return only the final sentence. No explanation, no quotes.\n\n"
                "Examples:\n"
                "HELLO DRINK WATER → 'Hello, please drink some water.'\n"
                "I HUNGRY → 'I am hungry.'\n"
                "YOU OKAY → 'Are you okay?'\n"
                "THANK YOU HELP → 'Thank you for helping me.'"
            )
        }
    ]

    # Add last 3 sentences as conversation context
    for entry in sentence_history[-3:]:
        messages.append({
            "role":    "assistant",
            "content": entry["sentence"]
        })

    # Current request — words listed clearly
    messages.append({
        "role":    "user",
        "content": f"Words signed: {', '.join(words)}. Make a grammatically correct sentence using only these words."
    })

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,  # very low = stays faithful to input words
            max_tokens=100,
        )
        sentence = response.choices[0].message.content.strip()
        print(f"[Groq] Generated: {sentence}")

    except Exception as e:
        print(f"[Groq] Error: {e}")
        question_words = {"WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "DO", "CAN", "IS", "ARE"}
        sentence = " ".join(w.capitalize() for w in words)
        sentence += "?" if any(w.upper() in question_words for w in words) else "."

    # Save to history
    sentence_history.append({
        "sentence":  sentence,
        "words":     words,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
    })
    if len(sentence_history) > MAX_HISTORY:
        sentence_history.pop(0)

    return {"sentence": sentence, "history": sentence_history}


# ── History endpoints ─────────────────────────────────────────────────────────
@app.get("/history")
def get_history():
    """Returns the last MAX_HISTORY generated sentences."""
    return {"history": sentence_history}


@app.delete("/history")
def clear_history():
    """Clears all sentence history. Called by frontend Clear History button."""
    sentence_history.clear()
    print("[History] Cleared")
    return {"status": "cleared", "history": sentence_history}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  True,
        "classes":       list(label_encoder.classes_),
        "sequence_len":  SEQUENCE_LENGTH,
        "history_count": len(sentence_history),
    }


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>index.html not found.</h2>"