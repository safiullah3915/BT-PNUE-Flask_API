from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

app = Flask(__name__)

# Load models
def load_and_compile_model(model_path, loss_function=None):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        if loss_function:
            model.compile(
                optimizer='adam',
                loss=loss_function,
                metrics=['accuracy']
            )
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")

pneumonia_model = load_and_compile_model('pneumonia_model.hdf5', SparseCategoricalCrossentropy(reduction='sum_over_batch_size'))
brain_tumor_model = load_and_compile_model('brain_tumor_model.h5')

def preprocess_image(image_bytes, model_name):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if model_name == 'pneumonia':
        image = ImageOps.fit(image, (180, 180), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    elif model_name == 'brain_tumor':
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.array(image) / 255.0
        img = preprocess_input(image_array)
    else:
        raise ValueError("Unsupported model")
    return np.expand_dims(img, axis=0)

@app.route('/classify_pneumonia', methods=['POST','GET'])
def classify_pneumonia():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image_bytes = file.read()
        image_array = preprocess_image(image_bytes, 'pneumonia')
        predictions = pneumonia_model.predict(image_array)
        class_index = np.argmax(predictions)
        class_labels = ['Normal', 'Pneumonia']  # Replace with your actual class labels
        return jsonify({'class': class_labels[class_index]})
    except Exception as e:
        return jsonify({'error': f'Error during classification: {e}'}), 500

@app.route('/classify_brain_tumor', methods=['POST','GET'])
def classify_brain_tumor():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image_bytes = file.read()
        image_array = preprocess_image(image_bytes, 'brain_tumor')
        predictions = brain_tumor_model.predict(image_array)
        class_label = 'Tumor' if predictions[0] > 0.5 else 'No Tumor'
        return jsonify({'class': class_label})
    except Exception as e:
        return jsonify({'error': f'Error during classification: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
