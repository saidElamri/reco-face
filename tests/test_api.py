from fastapi.testclient import TestClient
from main import app
from database import Base, engine, SessionLocal, get_db
import pytest
from sqlalchemy.orm import sessionmaker
import os

# Use the actual database for testing
TestingSessionLocal = SessionLocal

# Override get_db dependency for tests
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(name="client")
def client_fixture():
    # Create the tables in the test database
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as client:
        yield client
    # Drop the tables when tests are done
    Base.metadata.drop_all(bind=engine)

def test_predict_emotion_no_face_detected(client):
    # Create a dummy image file that won't have a face
    with open("test_no_face.jpg", "wb") as f:
        f.write(b"dummy image data") # This will not be a valid image but good enough to test no face detection

    with open("test_no_face.jpg", "rb") as f:
        response = client.post("/predict_emotion", files={"file": ("test_no_face.jpg", f, "image/jpeg")})
    
    assert response.status_code == 200
    assert response.json() == {"error": "No face detected"}

    # Clean up dummy file
    import os
    os.remove("test_no_face.jpg")


def test_get_history_empty(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == []
