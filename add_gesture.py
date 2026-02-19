import cv2
import time
import numpy as np
import mediapipe as mp
import pandas as pd
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


GESTURE_NAME = str(input("Enter gesture Name: "))
NUM_PHOTOS = 400
CSV_FILE = 'balanced_landmarks.csv'
MODEL_PATH = 'hand_landmarker.task'


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1, running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_coords(landmarks):
    
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    coords -= coords[0] 
    max_val = np.max(np.abs(coords))
    if max_val > 0: coords /= max_val
    return coords.flatten()

cap = cv2.VideoCapture(0)
new_rows = []
print(f"Recording {GESTURE_NAME}. You need to move your hand around to capture different orientations. Get ready...")
time.sleep(5)

while len(new_rows) < NUM_PHOTOS:
    ret, frame = cap.read()
    if not ret: break
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        norm_data = normalize_coords(result.hand_landmarks[0])
        new_rows.append([GESTURE_NAME] + norm_data.tolist())
        cv2.putText(frame, f"Captured: {len(new_rows)}/{NUM_PHOTOS}", (50,50), 1, 2, (0,255,0), 2)

    cv2.imshow('Recording', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

new_df = pd.DataFrame(new_rows, columns=pd.read_csv(CSV_FILE, nrows=1).columns)
updated_df = pd.concat([pd.read_csv(CSV_FILE), new_df])
updated_df.to_csv(CSV_FILE, index=False)
print("CSV Updated. Starting full retrain...")

from sklearn.preprocessing import LabelEncoder

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
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=40, batch_size=32, verbose=1) 

model.save('gesture_recognizer.keras')
np.save('classes.npy', le.classes_)
print(f"Done! Model now accurately knows: {list(le.classes_)}")