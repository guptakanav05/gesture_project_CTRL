import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


CSV_FILE = 'balanced_landmarks.csv'
GESTURE_TO_DELETE = "03_fist"

df = pd.read_csv(CSV_FILE)
original_count = len(df)

df_filtered = df[df['label'] != GESTURE_TO_DELETE]

print(f"Deleted {original_count - len(df_filtered)} rows of '{GESTURE_TO_DELETE}'.")

df_filtered.to_csv(CSV_FILE, index=False)

X = df_filtered.iloc[:, 1:].values.astype('float32')
le = LabelEncoder()
y = le.fit_transform(df_filtered.iloc[:, 0])
num_classes = len(le.classes_)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"Retraining model with remaining gestures: {list(le.classes_)}...")
model.fit(X, y, epochs=40, batch_size=32, verbose=1)

model.save('gesture_recognizer.keras')
np.save('classes.npy', le.classes_)

print("Done! The gesture has been completely erased from the system.")