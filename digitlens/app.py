"""Flask server: receives a canvas drawing, returns digit probabilities."""
import base64
import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from scipy import ndimage

from model import DigitCNN

MEAN, STD = 0.1307, 0.3081
MODEL_PATH = "model.pth"

app = Flask(__name__)
model = DigitCNN()
model_loaded = False
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model_loaded = True
else:
    print("WARNING: model.pth not found. Run `python train.py` first.")


def preprocess(data_url: str):
    """Turn a canvas PNG into a 1x1x28x28 tensor that looks like MNIST.

    MNIST digits are cropped, scaled to fit a 20x20 box, then placed in a
    28x28 image so that the center of mass sits at the middle. Doing the
    same here matters a lot for accuracy on hand-drawn input.
    """
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGBA")

    # Flatten onto black: the canvas is transparent where nothing is drawn.
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    gray = np.array(Image.alpha_composite(background, img).convert("L"), dtype=np.uint8)

    ys, xs = np.where(gray > 20)
    if len(xs) == 0:
        return None  # blank canvas

    gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = gray.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = np.array(Image.fromarray(gray).resize((new_w, new_h), Image.LANCZOS))

    canvas = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit

    # Shift so the center of mass is at (14, 14), like MNIST.
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant")

    x = torch.from_numpy(canvas.astype(np.float32) / 255.0)
    x = (x - MEAN) / STD
    return x.unsqueeze(0).unsqueeze(0), canvas


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify(error="Model not trained yet. Run `python train.py`, then restart the server."), 503

    payload = request.get_json(silent=True) or {}
    if "image" not in payload:
        return jsonify(error="No image received."), 400

    result = preprocess(payload["image"])
    if result is None:
        return jsonify(error="The canvas is empty. Draw a digit first."), 400
    tensor, processed = result

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].tolist()

    best = int(np.argmax(probs))
    return jsonify(
        digit=best,
        confidence=probs[best],
        probabilities=probs,
        # What the network actually sees (28x28, values 0-255).
        processed=processed.flatten().tolist(),
    )


if __name__ == "__main__":
    app.run(debug=True)
