from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('digit_recognition_model.h5')

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")

        file = request.files['file']

        # Open the image, convert to grayscale, resize to 28x28, and normalize
        img = Image.open(file).convert('L').resize((28, 28))
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # Predict the digit
        predicted_digit = np.argmax(model.predict(img_array))
        prediction = f"The predicted digit is: {predicted_digit}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
