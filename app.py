from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('digit_recognition_model.h5')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    # Open the image, convert to grayscale, resize to 28x28, and normalize
    img = Image.open(file).convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    # Predict the digit
    prediction = np.argmax(model.predict(img_array))

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
