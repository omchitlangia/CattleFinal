from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# -------------------------------
# Load Models
# -------------------------------
try:
    crop_model = joblib.load("crop_yield/crop_yield_model.joblib")
    print("✅ Crop Yield Model loaded")
except Exception as e:
    print("⚠️ Error loading crop model:", e)
    crop_model = None

try:
    milk_model = joblib.load("milk_quality/milk_quality_model.pkl")
    print("✅ Milk Quality Model loaded")
except Exception as e:
    print("⚠️ Error loading milk model:", e)
    milk_model = None

try:
    cattle_model = tf.keras.models.load_model("best_cattle/best_model_7class.h5")
    print("✅ Cattle Recognition Model loaded")
except Exception as e:
    print("⚠️ Error loading cattle model:", e)
    cattle_model = None

# -------------------------------
# ROUTES TO LOAD HTML PAGES
# -------------------------------
@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/milk")
def milk_page():
    return render_template("milk/milk.html")

@app.route("/crop")
def crop_page():
    return render_template("yield/crop-yield.html")

@app.route("/cattle")
def cattle_page():
    return render_template("cattle/cattle.html")

# -------------------------------
# API ROUTES
# -------------------------------
@app.route("/predict/crop", methods=["POST"])
def predict_crop():
    if not crop_model:
        return jsonify({"error": "Crop model not loaded"}), 500
    data = request.json
    try:
        features = np.array([list(data.values())], dtype=float)
        prediction = crop_model.predict(features)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/milk", methods=["POST"])
def predict_milk():
    if not milk_model:
        return jsonify({"error": "Milk model not loaded"}), 500
    data = request.json
    try:
        features = np.array([list(data.values())], dtype=float)
        prediction = milk_model.predict(features)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/cattle", methods=["POST"])
def predict_cattle():
    if not cattle_model:
        return jsonify({"error": "Cattle model not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file.stream).convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)
        prediction = cattle_model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
