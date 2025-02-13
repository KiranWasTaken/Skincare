from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("skin_type_model.h5")
class_labels = ["Dry", "Normal", "Oily"]

def preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        file_path = "uploads/" + file.filename
        file.save(file_path)
        processed_image = preprocess_image(file_path)
        predicted_class = class_labels[np.argmax(model.predict(processed_image))]
        return f"Predicted Skin Type: {predicted_class}"
    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
