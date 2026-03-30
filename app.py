import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras
from PIL import Image

app = Flask(__name__)

# paths
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/model.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["agriculture", "forest", "urban", "water"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# load model
print("loading model...")
model = keras.models.load_model(MODEL_PATH)
print("model loaded")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='choose a file first')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='file name is empty')

    try:
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        print("image uploaded")

        # get image
        img = Image.open(save_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype='float32')
        arr = np.expand_dims(arr, axis=0)

        # predict
        print("predicting...")
        pred = model.predict(arr, verbose=0)

        class_idx = np.argmax(pred[0])
        confidence = float(pred[0][class_idx]) * 100
        result = CLASS_NAMES[class_idx].capitalize()

        return render_template(
            'result.html',
            result=result,
            conf=round(confidence, 2),
            img_name=file.filename
        )

    except Exception as e:
        print("error:", e)
        return render_template('index.html', error='something went wrong')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
