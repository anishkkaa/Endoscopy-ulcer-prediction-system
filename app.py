import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.keras"
DATASET_DIR = BASE_DIR / "dataset"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

# -- Simple user store (in-memory) -------------------------------------------------
# NOTE: This is for demo purposes only. For production, use a database.
users = {}


def get_class_names():
    """Get class names from the dataset directory if available."""
    if DATASET_DIR.exists() and DATASET_DIR.is_dir():
        return sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
    # Fallback if dataset folder is not present.
    return ["AVM", "Normal", "Ulcer"]


CLASS_NAMES = get_class_names()


def load_model_safe():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training notebook to create best_model.keras")
    return tf.keras.models.load_model(str(MODEL_PATH))


model = None


def get_model():
    global model
    if model is None:
        model = load_model_safe()
    return model


# Set to True if your model was trained on images normalized to [0, 1].
# If the training code used raw 0-255 pixel values (as in the notebook), keep this False.
NORMALIZE_INPUT = False


def preprocess_image(image_path: str):
    img = image.load_img(image_path, target_size=(256, 256))
    x = image.img_to_array(img)
    if NORMALIZE_INPUT:
        x = x / 255.0
    return np.expand_dims(x, axis=0)


def predict_image(image_path: str):
    model = get_model()
    x = preprocess_image(image_path)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx),
        "confidence": float(probs[idx]),
        "all": {CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i): float(probs[i]) for i in range(len(probs))},
    }


# -- Authentication helpers -------------------------------------------------------

def is_authenticated():
    return session.get("username") is not None


def require_login():
    if not is_authenticated():
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))


# -- Routes -----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=("GET", "POST"))
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not password:
            flash("Username and password are required.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif username in users:
            flash("Username already exists. Please choose another.", "danger")
        else:
            users[username] = generate_password_hash(password)
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=("GET", "POST"))
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        stored_hash = users.get(username)
        if stored_hash and check_password_hash(stored_hash, password):
            session["username"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("index"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


@app.route("/predict", methods=("GET", "POST"))
def predict():
    prediction = None
    uploaded_filename = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part.", "danger")
            return redirect(request.url)

        f = request.files["image"]
        if f.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        uploads_dir = BASE_DIR / "static" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        uploaded_filename = f.filename
        file_path = uploads_dir / uploaded_filename
        f.save(str(file_path))

        try:
            prediction = predict_image(str(file_path))
        except Exception as ex:
            flash(f"Failed to predict: {ex}", "danger")

    return render_template(
        "predict.html",
        prediction=prediction,
        uploaded_filename=uploaded_filename,
        class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    app.run(debug=True)
