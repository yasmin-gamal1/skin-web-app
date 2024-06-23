### README

# Flask Image Classification Web App

This repository contains a simple Flask web application that serves a Keras deep learning model to classify images into two categories: "Benign" and "Malignant". The model is trained to detect whether an image represents a benign or malignant condition.

## Requirements

- Python 3.x
- Flask
- Keras
- TensorFlow
- NumPy
- Werkzeug

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-repo/flask-image-classification.git
cd flask-image-classification
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```

4. **Download and place the pre-trained Keras model:**

Ensure that you have the trained model file `new_model.h5` in the root directory of the project.

## Running the App

1. **Start the Flask app:**

```bash
python app.py
```

2. **Access the application:**

Open your web browser and navigate to `http://127.0.0.1:5001/`

## Project Structure

- `app.py`: The main Flask application file.
- `templates/index.html`: The main HTML file for the web interface.
- `new_model.h5`: The trained Keras model file (this needs to be in the root directory).
- `uploads/`: Directory where uploaded images are saved temporarily for processing.

## Usage

1. **Open the web application in your browser.**

2. **Upload an image file.**

3. **Click the "Predict" button to get the classification result.**

The result will indicate whether the image is classified as "Benign" or "Malignant".

## Code Explanation

### app.py

```python
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'new_model.h5'
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5001/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    return np.argmax(y)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        dic = {0: "Benign", 1: "Malignant"}
        pred_class = dic[preds]
        return pred_class
    return None

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

### templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Classification</title>
</head>
<body>
    <h1>Upload an image to classify</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Predict">
    </form>
</body>
</html>
```

## Notes

- Ensure that the uploaded images are in the correct format and size (224x224) for the model to process correctly.
- The current application is set up to run on port 5001. You can change this by modifying the `app.run()` call in `app.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

