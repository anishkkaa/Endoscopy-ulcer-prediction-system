# Endoscopy Classification Flask App

This repository contains a TensorFlow/Keras model for endoscopy image classification and a simple Flask app to run predictions via a web UI.

## Pages
- **Home** (`/`)
- **Register** (`/register`)
- **Login** (`/login`)
- **About** (`/about`)
- **Prediction** (`/predict`)

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open a browser at `http://127.0.0.1:5000`.

## Notes
- The user store is in-memory and resets when the app restarts.
- The model is loaded from `best_model.keras`.
- Uploaded images are stored under `static/uploads`.
