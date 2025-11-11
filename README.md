# Face Recognition and Emotion Detection

This project implements a real-time face recognition and emotion detection system. It utilizes deep learning models to identify faces and classify their emotions from live video streams or images.

## Features

- Real-time face detection
- Emotion classification (e.g., happy, sad, angry, neutral)
- Training script for emotion detection model
- Database integration for storing face data (potentially)

## Installation

To set up the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/face-reco.git
    cd face-reco
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Emotion Model

To train your own emotion detection model, run:

```bash
python train.py
```

### Running Live Face and Emotion Recognition

To start the live recognition system, run:

```bash
python live_reco.py
```

### Main Application

To run the main application (if different from live_reco), run:

```bash
python main.py
```

## Project Structure

-   `main.py`: Main entry point of the application.
-   `live_reco.py`: Script for real-time face and emotion recognition.
-   `train.py`: Script for training the emotion detection model.
-   `model.py`: Defines the deep learning model architecture.
-   `database.py`: Handles database operations (e.g., storing face encodings, user data).
-   `emotion_model.h5`: Pre-trained emotion detection model.
-   `emotion_model_improved.h5`: An improved version of the pre-trained emotion detection model.
-   `requirements.txt`: Lists all Python dependencies.
-   `EDA.ipynb`: Jupyter notebook for Exploratory Data Analysis.
-   `tests/`: Contains unit tests for the project.
-   `emotion_dataset/`: Directory for emotion training data.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
