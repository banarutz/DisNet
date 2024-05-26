from flask import Flask, request, jsonify
import onnxruntime
import numpy as np
from PIL import Image
import torch
import cv2

app = Flask(__name__)

# Load the ONNX model
onnx_model_path = "/home/sbanaru/Desktop/DisNet/model_zoo/resnet18_acc87.6.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) /255.0
    image = np.transpose(image, (2, 0, 1))  # Change from HWC to CHW format
    image = np.expand_dims(image, axis=0)
    print(np.min(image), np.max(image))
    return image.astype(np.float32)

# Function to perform inference
def predict_class_and_confidence(image_path):
    input_data = preprocess_image(image_path)
    outputs = ort_session.run(None, {"input.1": input_data})
    scores = np.squeeze(outputs[0])
    predicted_class = np.argmax(scores)
    confidence = torch.max(torch.nn.functional.softmax(torch.Tensor(outputs[0]), dim=1)[0] * 100).item()
    return predicted_class, confidence

# Endpoint for performing inference
@app.route('/predict', methods=['POST'])
def predict():
    if 'image_path' not in request.files:
        return jsonify({'error': 'No image path provided'})
    
    image_file = request.files['image_path']
    image_path = "temp_image.jpg"
    image_file.save(image_path)
    
    predicted_class, confidence = predict_class_and_confidence(image_path)
    class_labels = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British', 'Egyptian', 'Maine', 'Persian', 'Ragdoll', 'Russian', 'Siamese', 'Sphynx', 'american', 'basset', 'beagle', 'boxer', 'chihuahua', 'english', 'german', 'great', 'havanese', 'japanese', 'keeshond', 'leonberger', 'miniature', 'newfoundland', 'pomeranian', 'pug', 'saint', 'samoyed', 'scottish', 'shiba', 'staffordshire', 'wheaten', 'yorkshire']
    predicted_class = class_labels[predicted_class]

    return jsonify({'predicted_class': str(predicted_class), 'confidence': str(confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969)
