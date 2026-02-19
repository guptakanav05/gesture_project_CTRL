"""
CTRL — Gesture Control Backend
Flask API server that bridges the frontend dashboard with the
gesture recognition, add-gesture, and delete-gesture backends.

Run:  python app.py
Open: http://127.0.0.1:5000
"""

import os
import shutil
import threading
import time
import queue
import json
from datetime import datetime
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import subprocess
import platform

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'gesture_recognizer.keras')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.npy')
TASK_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')
CSV_FILE = os.path.join(BASE_DIR, 'balanced_landmarks.csv')
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')
MAX_BACKUPS = 5

# ─── Backup Helper ───────────────────────────────────────
def backup_training_data(reason="manual"):
    """Create a timestamped backup of CSV, model, and classes.npy.
    Keeps at most MAX_BACKUPS, rotating out the oldest.
    Returns the backup folder path or None if nothing to back up."""
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Only back up if at least the CSV exists
    if not os.path.exists(CSV_FILE):
        return None

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(BACKUP_DIR, f'{stamp}_{reason}')
    os.makedirs(folder, exist_ok=True)

    for src in [CSV_FILE, MODEL_PATH, CLASSES_PATH]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(folder, os.path.basename(src)))

    # Rotate: keep only the newest MAX_BACKUPS
    existing = sorted(
        [d for d in os.listdir(BACKUP_DIR)
         if os.path.isdir(os.path.join(BACKUP_DIR, d))],
        reverse=True
    )
    for old in existing[MAX_BACKUPS:]:
        shutil.rmtree(os.path.join(BACKUP_DIR, old), ignore_errors=True)

    print(f"[BACKUP] Saved → {folder}")
    return folder


# ─── Flask App ───────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ─── Shared State ────────────────────────────────────────
recognition_thread = None
recognition_running = False
gesture_events = queue.Queue(maxsize=100)
background_task = {
    "running": False,
    "type": None,
    "gesture": None,
    "progress": "",
    "error": None,
}

# Add-gesture recording state
add_gesture_cancelled = False
recording_frame = None          # latest JPEG bytes for MJPEG stream
recording_frame_lock = threading.Lock()
last_gesture_completed_at = 0   # timestamp of last recording completion
GESTURE_COOLDOWN_SECONDS = 6    # min seconds between consecutive recordings

# ═══════════════════════════════════════════════════════════
#  STATIC FILE SERVING
# ═══════════════════════════════════════════════════════════

@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/style.css')
def serve_css():
    return send_from_directory(BASE_DIR, 'style.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory(BASE_DIR, 'script.js')

# ═══════════════════════════════════════════════════════════
#  API — GESTURES
# ═══════════════════════════════════════════════════════════

@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    """Return known gesture classes from classes.npy"""
    try:
        if os.path.exists(CLASSES_PATH):
            import re
            classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
            # Strip numeric prefix (e.g. '01_palm' → 'palm')
            clean = [re.sub(r'^\d+_', '', c) for c in classes]
            return jsonify({"gestures": clean})
        return jsonify({"gestures": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════
#  API — RECOGNITION START / STOP / STREAM
# ═══════════════════════════════════════════════════════════

def normalize_landmarks(landmarks):
    """Same normalization used during training."""
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    coords = coords - coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten().reshape(1, -1)


# ═══════════════════════════════════════════════════════════
#  ACTION EXECUTION
# ═══════════════════════════════════════════════════════════

def execute_action(action_id):
    """Execute a desktop action based on its ID. Returns action name or None."""
    import pyautogui
    pyautogui.FAILSAFE = False

    IS_MAC = platform.system() == 'Darwin'
    # modifier key: 'command' on macOS, 'ctrl' on Windows/Linux
    MOD = 'command' if IS_MAC else 'ctrl'

    try:
        if action_id == 'a1':  # Switch Tab Right
            pyautogui.hotkey(MOD, 'tab') if not IS_MAC else pyautogui.hotkey('command', 'option', 'right')
            return 'Switch Tab Right'

        elif action_id == 'a2':  # Switch Tab Left
            pyautogui.hotkey(MOD, 'shift', 'tab') if not IS_MAC else pyautogui.hotkey('command', 'option', 'left')
            return 'Switch Tab Left'

        elif action_id == 'a3':  # Open New Tab
            pyautogui.hotkey(MOD, 't')
            return 'Open New Tab'

        elif action_id == 'a4':  # Close Tab
            pyautogui.hotkey(MOD, 'w')
            return 'Close Tab'

        elif action_id == 'a5':  # Play / Pause Video
            pyautogui.press('space')
            return 'Play / Pause Video'

        elif action_id == 'a6':  # Mute / Unmute
            if IS_MAC:
                # Use AppleScript — pyautogui F-keys don't send media keys
                subprocess.run(['osascript', '-e',
                    'set volume output muted (not (output muted of (get volume settings)))'])
            else:
                pyautogui.hotkey('volumemute')
            return 'Mute / Unmute'

        elif action_id == 'a7':  # Open Calculator
            if IS_MAC:
                subprocess.Popen(['open', '-a', 'Calculator'])
            else:
                subprocess.Popen(['calc'])
            return 'Open Calculator'

        elif action_id == 'a8':  # Take Screenshot
            if IS_MAC:
                pyautogui.hotkey('command', 'shift', '3')
            else:
                pyautogui.hotkey('win', 'printscreen')
            return 'Take Screenshot'

        elif action_id == 'a9':  # Volume Up
            if IS_MAC:
                subprocess.run(['osascript', '-e',
                    'set volume output volume ((output volume of (get volume settings)) + 10)'])
            else:
                pyautogui.hotkey('volumeup')
            return 'Volume Up'

        elif action_id == 'a10':  # Volume Down
            if IS_MAC:
                subprocess.run(['osascript', '-e',
                    'set volume output volume ((output volume of (get volume settings)) - 10)'])
            else:
                pyautogui.hotkey('volumedown')
            return 'Volume Down'

        elif action_id == 'a11':  # Refresh Page
            pyautogui.hotkey(MOD, 'r')
            return 'Refresh Page'

        elif action_id == 'a12':  # Energy Saving Mode
            if IS_MAC:
                subprocess.Popen(['pmset', 'displaysleepnow'])
            else:
                subprocess.Popen(['rundll32.exe', 'powrprof.dll,SetSuspendState', '0', '1', '0'])
            return 'Energy Saving Mode'

        elif action_id == 'a13':  # Shut Down PC
            if IS_MAC:
                subprocess.Popen(['osascript', '-e',
                    'tell app "System Events" to shut down'])
            else:
                subprocess.Popen(['shutdown', '/s', '/t', '5'])
            return 'Shut Down PC'

        elif action_id == 'a14':  # Lock Screen
            if IS_MAC:
                pyautogui.hotkey('command', 'ctrl', 'q')
            else:
                subprocess.Popen(['rundll32.exe', 'user32.dll,LockWorkStation'])
            return 'Lock Screen'

    except Exception as e:
        print(f"[Action Error] {action_id}: {e}")
    return None


ACTION_COOLDOWN = 6.0  # seconds between repeated action triggers
last_action_time = 0


def recognition_loop():
    """Background thread: webcam → hand-landmarker → model → execute action → SSE."""
    global recognition_running, last_action_time
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = np.load(CLASSES_PATH, allow_pickle=True)

        base_options = mp_python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        timestamp = 0

        while recognition_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp += 1
            result = detector.detect_for_video(mp_image, timestamp)

            if result.hand_landmarks:
                input_data = normalize_landmarks(result.hand_landmarks[0])
                prediction = model.predict(input_data, verbose=0)
                class_id = np.argmax(prediction)
                confidence = float(prediction[0][class_id])
                gesture_name = str(class_names[class_id])

                if confidence > 0.8:
                    # Strip numeric prefix (e.g. '01_palm' → 'palm')
                    import re
                    clean_gesture = re.sub(r'^\d+_', '', gesture_name)
                    event_data = {
                        "gesture": clean_gesture,
                        "confidence": round(confidence * 100),
                        "timestamp": time.time(),
                        "action": None,
                    }

                    # Always look up mappings to populate action name
                    mappings = load_mappings()
                    matched_action_id = None
                    # Strip numeric prefix (e.g. '01_palm' → 'palm')
                    import re
                    clean_name = re.sub(r'^\d+_', '', gesture_name).lower()
                    for m in mappings:
                        gid = m.get('gestureId', '').lower()
                        if gid == clean_name or gid == gesture_name.lower():
                            matched_action_id = m['actionId']
                            break

                    if matched_action_id:
                        # Execute only if cooldown has elapsed
                        now = time.time()
                        if now - last_action_time >= ACTION_COOLDOWN:
                            action_name = execute_action(matched_action_id)
                            if action_name:
                                event_data["action"] = action_name
                                last_action_time = now
                                print(f"[Action] {gesture_name} → {action_name}")
                        else:
                            # Cooldown active — still show what action is mapped
                            action_label = matched_action_id
                            event_data["action"] = f"{action_label} (cooldown)"
                    else:
                        if mappings:
                            print(f"[Recognition] No mapping match for '{gesture_name}' in {[m.get('gestureId') for m in mappings]}")

                    try:
                        gesture_events.put_nowait(event_data)
                    except queue.Full:
                        try:
                            gesture_events.get_nowait()
                        except queue.Empty:
                            pass
                        gesture_events.put_nowait(event_data)

            time.sleep(0.033)  # ~30 FPS

        cap.release()
    except Exception as e:
        print(f"[Recognition Error] {e}")
    finally:
        recognition_running = False


@app.route('/api/recognize/start', methods=['POST'])
def start_recognition():
    global recognition_thread, recognition_running

    if recognition_running:
        return jsonify({"status": "already_running"})

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        return jsonify({"error": "Model files not found. Train a model first."}), 400

    recognition_running = True
    recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
    recognition_thread.start()
    return jsonify({"status": "started"})


@app.route('/api/recognize/stop', methods=['POST'])
def stop_recognition():
    global recognition_running
    recognition_running = False
    return jsonify({"status": "stopped"})


@app.route('/api/recognize/stream')
def recognition_stream():
    """SSE endpoint — streams detected gestures to the frontend in real-time."""
    def generate():
        while True:
            try:
                event = gesture_events.get(timeout=1)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "recognition": recognition_running,
        "backgroundTask": background_task,
    })

# ═══════════════════════════════════════════════════════════
#  API — ADD GESTURE
# ═══════════════════════════════════════════════════════════

def add_gesture_worker(gesture_name, num_photos=400):
    """Background: record hand landmarks via webcam → append CSV → retrain."""
    import tensorflow as tf
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    global background_task, add_gesture_cancelled, recording_frame
    add_gesture_cancelled = False
    background_task.update(running=True, type="add", gesture=gesture_name,
                           progress="Initializing camera...", error=None)
    try:
        base_options = mp_python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        def normalize_coords(landmarks):
            coords = np.array([[lm.x, lm.y] for lm in landmarks])
            coords -= coords[0]
            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords /= max_val
            return coords.flatten()

        cap = cv2.VideoCapture(0)
        new_rows = []
        background_task["progress"] = f"Recording '{gesture_name}'... show your hand"

        while len(new_rows) < num_photos:
            # Check cancel flag
            if add_gesture_cancelled:
                cap.release()
                with recording_frame_lock:
                    recording_frame = None
                background_task.update(
                    progress=f"Cancelled recording '{gesture_name}'",
                    error="Cancelled by user", running=False)
                return

            ret, frame = cap.read()
            if not ret:
                break

            # Draw status overlay on the frame
            display = frame.copy()
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = detector.detect(mp_image)
            hand_detected = False
            if result.hand_landmarks:
                hand_detected = True
                norm_data = normalize_coords(result.hand_landmarks[0])
                new_rows.append([gesture_name] + norm_data.tolist())
                # Draw hand landmarks on display frame
                for lm in result.hand_landmarks[0]:
                    h, w = display.shape[:2]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)

            # Status text on frame
            status_text = f"Captured: {len(new_rows)}/{num_photos}"
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if not hand_detected:
                cv2.putText(display, "Show your hand...", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

            # Encode frame as JPEG and store for MJPEG streaming
            _, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with recording_frame_lock:
                recording_frame = jpeg.tobytes()

            background_task["progress"] = f"Captured {len(new_rows)}/{num_photos}"

        cap.release()
        with recording_frame_lock:
            recording_frame = None

        # ── Cancel gate 2: after recording, before touching files ──
        if add_gesture_cancelled:
            background_task.update(
                progress=f"Cancelled recording '{gesture_name}' — no data saved",
                error="Cancelled by user", running=False)
            return

        if not new_rows:
            background_task.update(progress="No hand detected", error="No data captured", running=False)
            return

        # Build / update CSV
        background_task["progress"] = "Backing up existing data..."
        backup_folder = backup_training_data(reason=f"before_add_{gesture_name}")

        # ── Cancel gate 3: after backup, before CSV write ──
        if add_gesture_cancelled:
            background_task.update(
                progress=f"Cancelled '{gesture_name}' — no data saved",
                error="Cancelled by user", running=False)
            return

        background_task["progress"] = "Updating dataset..."
        columns = ['label'] + [f'x{i}' for i in range(42)]
        
        try:
            if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
                existing = pd.read_csv(CSV_FILE)
                # Ensure columns match, or reset if mismatched? For now assume match or just append.
                # If existing has no columns, it's effectively empty.
                if existing.empty:
                     existing = pd.DataFrame(columns=columns)
                else:
                     columns = existing.columns.tolist()
            else:
                existing = pd.DataFrame(columns=columns)
        except Exception:
            # If any error reading (corrupt/empty), start fresh
            existing = pd.DataFrame(columns=columns)

        new_df = pd.DataFrame(new_rows, columns=columns)
        updated_df = pd.concat([existing, new_df], ignore_index=True)
        updated_df.to_csv(CSV_FILE, index=False)

        # ── Cancel gate 4: after CSV write — rollback from backup ──
        if add_gesture_cancelled:
            background_task["progress"] = "Cancelled — rolling back data..."
            if backup_folder:
                for filename in os.listdir(backup_folder):
                    shutil.copy2(
                        os.path.join(backup_folder, filename),
                        os.path.join(BASE_DIR, filename))
            background_task.update(
                progress=f"Cancelled '{gesture_name}' — data rolled back",
                error="Cancelled by user", running=False)
            return

        # Retrain
        background_task["progress"] = "Retraining model... this may take a minute"
        df = pd.read_csv(CSV_FILE)
        X = df.iloc[:, 1:].values.astype('float32')
        le = LabelEncoder()
        y = le.fit_transform(df.iloc[:, 0])
        num_classes = len(le.classes_)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(42,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=40, batch_size=32, verbose=0)

        # ── Cancel gate 5: after retrain — rollback everything ──
        if add_gesture_cancelled:
            background_task["progress"] = "Cancelled — rolling back data and model..."
            if backup_folder:
                for filename in os.listdir(backup_folder):
                    shutil.copy2(
                        os.path.join(backup_folder, filename),
                        os.path.join(BASE_DIR, filename))
            background_task.update(
                progress=f"Cancelled '{gesture_name}' — fully rolled back",
                error="Cancelled by user", running=False)
            return

        model.save(MODEL_PATH)
        np.save(CLASSES_PATH, le.classes_)
        background_task["progress"] = f"Done! '{gesture_name}' added successfully"
    except Exception as e:
        background_task["error"] = str(e)
        background_task["progress"] = f"Error: {e}"
    finally:
        with recording_frame_lock:
            recording_frame = None
        background_task["running"] = False
        global last_gesture_completed_at
        last_gesture_completed_at = time.time()


@app.route('/api/gesture/add', methods=['POST'])
def add_gesture():
    global background_task, add_gesture_cancelled
    if background_task["running"]:
        return jsonify({"error": "A background task is already running"}), 409

    # Enforce cooldown between recordings
    elapsed = time.time() - last_gesture_completed_at
    if elapsed < GESTURE_COOLDOWN_SECONDS:
        remaining = int(GESTURE_COOLDOWN_SECONDS - elapsed)
        return jsonify({"error": f"Please wait {remaining}s before recording another gesture"}), 429

    data = request.get_json(force=True)
    gesture_name = data.get("name", "").strip()
    if not gesture_name:
        return jsonify({"error": "Gesture name is required"}), 400

    add_gesture_cancelled = False
    thread = threading.Thread(target=add_gesture_worker, args=(gesture_name,), daemon=True)
    thread.start()
    return jsonify({"status": "recording_started", "gesture": gesture_name})


@app.route('/api/gesture/add/cancel', methods=['POST'])
def cancel_add_gesture():
    global add_gesture_cancelled
    add_gesture_cancelled = True
    return jsonify({"status": "cancel_requested"})


@app.route('/api/gesture/add/stream')
def add_gesture_stream():
    """MJPEG stream of the webcam during gesture recording."""
    def generate():
        while True:
            with recording_frame_lock:
                frame = recording_frame
            if frame is None:
                # No frame yet or recording finished — send a blank
                time.sleep(0.05)
                # Check if recording is still running
                if not background_task.get("running") or background_task.get("type") != "add":
                    break
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache'})

# ═══════════════════════════════════════════════════════════
#  API — DELETE GESTURE
# ═══════════════════════════════════════════════════════════

def delete_gesture_worker(gesture_name):
    """Background: remove from CSV → retrain."""
    import tensorflow as tf
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    global background_task
    background_task.update(running=True, type="delete", gesture=gesture_name,
                           progress=f"Deleting '{gesture_name}'...", error=None)
    try:
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            background_task.update(progress="Dataset is empty", error="No data to delete", running=False)
            return

        # Backup before delete
        background_task["progress"] = "Backing up existing data..."
        backup_training_data(reason=f"before_delete_{gesture_name}")

        try:
            df = pd.read_csv(CSV_FILE)
        except Exception:
             background_task.update(progress="Dataset corrupted/empty", error="Cannot read CSV", running=False)
             return
        df_filtered = df[df.iloc[:, 0] != gesture_name]
        removed = len(df) - len(df_filtered)

        if removed == 0:
            background_task.update(progress=f"'{gesture_name}' not found in dataset",
                                   error="Gesture not found", running=False)
            return

        df_filtered.to_csv(CSV_FILE, index=False)
        background_task["progress"] = f"Removed {removed} samples. Retraining..."

        X = df_filtered.iloc[:, 1:].values.astype('float32')
        le = LabelEncoder()
        y = le.fit_transform(df_filtered.iloc[:, 0])
        num_classes = len(le.classes_)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(42,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=40, batch_size=32, verbose=0)

        model.save(MODEL_PATH)
        np.save(CLASSES_PATH, le.classes_)
        background_task["progress"] = f"Done! '{gesture_name}' removed successfully"
    except Exception as e:
        background_task["error"] = str(e)
        background_task["progress"] = f"Error: {e}"
    finally:
        background_task["running"] = False


# ─── Backup Management Endpoints ─────────────────────────
@app.route('/api/backups', methods=['GET'])
def list_backups():
    """List available training data backups."""
    if not os.path.exists(BACKUP_DIR):
        return jsonify({"backups": []})
    backups = []
    for name in sorted(os.listdir(BACKUP_DIR), reverse=True):
        path = os.path.join(BACKUP_DIR, name)
        if os.path.isdir(path):
            files = os.listdir(path)
            backups.append({"name": name, "files": files})
    return jsonify({"backups": backups})


@app.route('/api/backups/restore', methods=['POST'])
def restore_backup():
    """Restore training data from a named backup."""
    data = request.get_json(force=True)
    backup_name = data.get("name", "").strip()
    if not backup_name:
        return jsonify({"error": "Backup name is required"}), 400

    backup_path = os.path.join(BACKUP_DIR, backup_name)
    if not os.path.isdir(backup_path):
        return jsonify({"error": f"Backup '{backup_name}' not found"}), 404

    try:
        # Restore each file
        restored = []
        for filename in os.listdir(backup_path):
            src = os.path.join(backup_path, filename)
            dst = os.path.join(BASE_DIR, filename)
            shutil.copy2(src, dst)
            restored.append(filename)
        return jsonify({"success": True, "restored": restored,
                        "message": f"Restored {len(restored)} files from '{backup_name}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/gesture/delete', methods=['POST'])
def delete_gesture():
    global background_task
    if background_task["running"]:
        return jsonify({"error": "A background task is already running"}), 409

    data = request.get_json(force=True)
    gesture_name = data.get("name", "").strip()
    if not gesture_name:
        return jsonify({"error": "Gesture name is required"}), 400

    thread = threading.Thread(target=delete_gesture_worker, args=(gesture_name,), daemon=True)
    thread.start()
    return jsonify({"status": "delete_started", "gesture": gesture_name})

# ═══════════════════════════════════════════════════════════
#  API — MAPPINGS PERSISTENCE
# ═══════════════════════════════════════════════════════════

MAPPINGS_FILE = os.path.join(BASE_DIR, 'mappings.json')

def load_mappings():
    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_mappings_file(mappings):
    with open(MAPPINGS_FILE, 'w') as f:
        json.dump(mappings, f, indent=2)

@app.route('/api/mappings', methods=['GET'])
def get_mappings():
    m = load_mappings()
    print(f"[Mappings GET] Returning {len(m)} mappings: {m}")
    return jsonify({"mappings": m})

@app.route('/api/mappings', methods=['PUT'])
def replace_all_mappings():
    """Replace the entire mappings list — frontend is the source of truth."""
    data = request.get_json(force=True)
    mappings = data if isinstance(data, list) else data.get("mappings", [])
    print(f"[Mappings PUT] Received {len(mappings)} mappings: {mappings}")
    save_mappings_file(mappings)
    print(f"[Mappings PUT] Saved to {MAPPINGS_FILE}")
    return jsonify({"status": "ok", "count": len(mappings)})

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  CTRL — Gesture Control Server")
    print("  Open: http://127.0.0.1:5001")
    print("=" * 50 + "\n")
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
