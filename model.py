import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load model
MODEL_PATH = "emotion_model_improved.h5"
model = load_model(MODEL_PATH)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Corrected emotion labels based on the dataset
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def predict_emotion(image):
    """
    Predicts the emotion from an image.
    
    Args:
        image: A numpy array representing the image.
        
    Returns:
        A tuple containing the predicted label and confidence, or (None, None) if no face is detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    # Take the first face found
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
    roi_gray = roi_gray.astype("float") / 255.0
    roi_gray = img_to_array(roi_gray)
    roi_gray = np.expand_dims(roi_gray, axis=0)

    preds = model.predict(roi_gray)
    label = emotion_labels[np.argmax(preds)]
    confidence = float(np.max(preds))

    return label, confidence
