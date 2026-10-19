# DigitLens – Handwritten Digit Recognition

A deep learning mini project: a convolutional neural network (PyTorch) trained on
MNIST, served by Flask, with a drawing-canvas web frontend that predicts live as you draw.

## Project structure

```
digitlens/
├── model.py            CNN architecture
├── train.py            Downloads MNIST, trains, saves model.pth
├── app.py              Flask server + image preprocessing + /predict API
├── requirements.txt
├── templates/
│   └── index.html      Page markup
└── static/
    ├── style.css       Styling
    └── script.js       Canvas drawing + calls /predict
```

## Run it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py                   # ~2-5 min on CPU, reaches ~99% test accuracy
python app.py                     # open http://127.0.0.1:5000
```

## How it works

1. **Model** – three Conv→BatchNorm→ReLU blocks (32/64/128 filters), global average
   pooling, dropout, and a 10-way linear layer.
2. **Training** – Adam, StepLR schedule, random rotation/shift/scale augmentation so the
   model copes with hand-drawn variation. Cross-entropy loss.
3. **Preprocessing** (`app.py`) – the canvas drawing is cropped to its bounding box,
   scaled to fit 20×20, padded to 28×28 and shifted so its center of mass is centered.
   This mimics how MNIST was built and greatly improves accuracy on real drawings.
4. **API** – `POST /predict` with `{image: <PNG data URL>}` returns the predicted digit,
   confidence, all 10 probabilities and the 28×28 image the network saw.

## Ideas to extend

- Add Grad-CAM to highlight which strokes drove the prediction.
- Swap MNIST for EMNIST to recognize letters.
- Export to ONNX and run the model fully in the browser with ONNX Runtime Web.
