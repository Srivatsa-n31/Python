import os
from flask import Flask, request, jsonify, render_template
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\Srivatsa n\Cgproject\imageproject\model.h5'
model = load_model(model_path)

# Create the 'uploads' directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def predict_image(model, image_path):
    try:
        image = cv.imread(image_path)
        if image is None:
            return "Error: Unable to read image."
        resized_image = cv.resize(image, (32, 32))
        normalized_image = resized_image / 255.0
        normalized_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
        predicted_array = model.predict(normalized_image)
        predicted_class = "Living" if predicted_array[0][0] > 0.5 else "Non-living"
        return predicted_class
    except Exception as e:
        return f"Error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    prediction = predict_image(model, file_path)
    return jsonify({'prediction': prediction, 'image_path': file_path})

if __name__ == '__main__':
    app.run(debug=True)
