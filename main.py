from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
import numpy as np
import cv2

from database import get_db, Prediction
from model import predict_emotion

app = FastAPI()

@app.post("/predict_emotion")
async def predict_emotion_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    This endpoint receives an image, predicts the emotion, and saves the result to the database.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    label, confidence = predict_emotion(img)

    if label is None:
        return {"error": "No face detected"}

    # Save to DB
    pred = Prediction(emotion=label, confidence=confidence)
    db.add(pred)
    db.commit()
    db.refresh(pred)

    return {"emotion": label, "confidence": confidence}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    """
    This endpoint returns the history of all predictions.
    """
    results = db.query(Prediction).all()
    return [{"id": r.id, "emotion": r.emotion, "confidence": r.confidence, "created_at": r.created_at} for r in results]
