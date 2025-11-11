# 🎭 Face Recognition & Emotion Detection API

A comprehensive real-time face recognition and emotion detection system built with FastAPI, TensorFlow, and OpenCV. This project provides both a web API for emotion prediction and a live webcam-based emotion detection application.

![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)

## 🌟 Features

- **Real-time Emotion Detection**: Detect emotions from live webcam feed
- **RESTful API**: FastAPI-based endpoints for emotion prediction
- **Database Integration**: PostgreSQL storage for prediction history
- **Deep Learning Model**: CNN-based emotion classification with 7 emotion categories
- **Training Pipeline**: Custom model training with data augmentation
- **Easy Setup**: Docker support and comprehensive documentation

## 📊 Emotion Categories

The system recognizes 7 basic emotions:
- 😠 Angry
- 🤢 Disgusted
- 😨 Fearful
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprised

## 🏗️ Architecture

```
face-reco/
├── main.py              # FastAPI application & endpoints
├── model.py             # Emotion prediction logic
├── live_reco.py         # Live webcam detection
├── train.py             # Model training script
├── database.py          # PostgreSQL database setup
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── tests/              # Unit tests
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 13+
- OpenCV
- TensorFlow

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face-reco.git
   cd face-reco
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL database**
   ```bash
   # Create database
   createdb emotion_detection_db

   # Set environment variables
   export DB_USERNAME=your_username
   export DB_PASSWORD=your_password
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=emotion_detection_db
   ```

## 🎯 Usage

### 1. Run API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API Documentation**: http://localhost:8000/docs
**Alternative Docs**: http://localhost:8000/redoc

### 2. Live Emotion Detection

```bash
python live_reco.py
```

**Controls**: Press 'q' to quit the application.

### 3. Train Custom Model

```bash
python train.py
```

**Dataset Requirements**: Place emotion dataset in `emotion_dataset/` with subdirectories for each emotion category.

## 🔌 API Endpoints

### Predict Emotion

**POST** `/predict_emotion`

Upload an image to get emotion prediction.

**Request**:
```bash
curl -X POST "http://localhost:8000/predict_emotion" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

**Response**:
```json
{
  "emotion": "happy",
  "confidence": 0.92
}
```

### Get Prediction History

**GET** `/history`

Retrieve all past predictions.

**Response**:
```json
[
  {
    "id": 1,
    "emotion": "happy",
    "confidence": 0.92,
    "created_at": "2023-12-01T10:30:00.000Z"
  }
]
```

## 🧠 Model Details

### CNN Architecture
- **Input**: 48x48 pixel grayscale face images
- **Architecture**: Convolutional layers → Pooling → Dense layers
- **Output**: 7 emotion categories with softmax activation
- **Training**: Data augmentation with flipping, rotation, zooming

### Face Detection
- **Algorithm**: Haar Cascade Classifier
- **Preprocessing**: Grayscale conversion, normalization
- **Detection**: Multi-scale face detection

## 🗄️ Database Schema

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    emotion VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_model.py
```

## 📁 Project Structure

```
face-reco/
├── main.py              # FastAPI application
├── model.py             # Prediction logic and model loading
├── live_reco.py         # Real-time webcam detection
├── train.py             # Model training pipeline
├── database.py          # Database setup and ORM
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── tests/              # Unit tests
│   ├── test_model.py   # Model testing
│   └── test_api.py     # API testing
├── emotion_dataset/     # Training data directory
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── emotion_model_improved.h5  # Trained model file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_USERNAME` | PostgreSQL username | - |
| `DB_PASSWORD` | PostgreSQL password | - |
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | PostgreSQL database name | `emotion_detection_db` |

### Model Configuration

- **Image Size**: 48x48 pixels
- **Face Detection**: Haar Cascade with scaleFactor=1.3, minNeighbors=5
- **Confidence Threshold**: No threshold (returns all predictions)

## 🚀 Deployment

### Docker Support

```bash
# Build image
docker build -t face-reco .

# Run container
docker run -p 8000:8000 face-reco
```

### Production Considerations

- Use a production-grade database (PostgreSQL)
- Implement proper error handling and logging
- Add authentication and rate limiting
- Use a reverse proxy (nginx) for SSL termination
- Set up proper monitoring and alerts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

1. **No face detected**
   - Ensure good lighting conditions
   - Check camera permissions
   - Verify face is clearly visible

2. **Model loading error**
   - Check if `emotion_model_improved.h5` exists
   - Verify TensorFlow installation
   - Check model file permissions

3. **Database connection issues**
   - Verify PostgreSQL is running
   - Check environment variables
   - Ensure database exists

4. **API not accessible**
   - Check if port 8000 is available
   - Verify firewall settings
   - Check host configuration

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://opencv.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/)

---

Made with ❤️ by your name | Last updated: December 2023
