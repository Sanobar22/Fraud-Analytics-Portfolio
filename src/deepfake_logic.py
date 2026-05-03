import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image

# Path to save artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'deepfake_cnn_model.keras')

def build_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_images_from_folders(base_path, limit=500):
    images = []
    labels = []
    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(base_path, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))][:limit]
        for f in files:
            img = Image.open(os.path.join(folder_path, f)).convert('RGB')
            img = img.resize((128, 128))
            images.append(np.array(img) / 255.0)
            labels.append(label)
    return np.array(images), np.array(labels)

def train_deepfake_model(dataset_path):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    X, y = load_images_from_folders(dataset_path)
    model = build_model()
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)
    return "Deepfake CNN model trained successfully."

def predict_deepfake(pil_img):
    if not os.path.exists(MODEL_PATH):
        return None, "Model not trained yet."
    
    model = tf.keras.models.load_model(MODEL_PATH)
    img = pil_img.convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    probability = model.predict(img_array)[0][0]
    prediction = 1 if probability > 0.5 else 0
    return prediction, float(probability)
