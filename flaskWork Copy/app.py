from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import  load_model
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model(r'C:\Users\User\Desktop\flaskWork Copy\RGBmodel.keras')


def preprocess_image(image):
    
    image = image.resize((224, 224))
    #image = np.array(image).astype('float32') / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(model, image):
    # Preprocess the image
    image = preprocess_image(image)
    
    raw_prediction = model.predict(image)[0]
    print("Raw prediction:", raw_prediction)
    # Apply softmax to get probabilities
    prediction = tf.nn.softmax(raw_prediction)
    print("prediction:", prediction)
    #prediction = raw_prediction
    cat=["ยุคสมัยสุโขทัย","ยุคสมัยอยุธยาตอนต้น","ยุคสมัยอยุธยาตอนกลาง","ยุคสมัยอยุธยาตอนปลาย","ยุคสมัยรัตนโกสินทร์","ภาพอื่นๆ"]
    # Create the JSON response
    response = {
        "predicted_result": [
            {"number": cat[i], "probability": float(f"{prediction[i] * 100:.2f}")} for i in range(6)
        ]
    }
    
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Read the image
        image = Image.open(io.BytesIO(file.read()))
        
        # Predict the image
        response = predict_image(model, image)
        
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
