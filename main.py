from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
import logging
import threading

# Initialize Flask app
app = Flask(__name__, template_folder='.')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
model = load_model('facial_emotion_detection_model.h5')
predict_lock = threading.Lock()

# Define class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Emotion detection function
def detect_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Direct forward pass is more reliable than model.predict() under some server runtimes.
    with predict_lock:
        prediction = model(img_array, training=False).numpy()
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)

    return predicted_class, confidence

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        started = time.time()
        logger.info('POST / received for inference')
        if 'file' not in request.files:
            return 'No file uploaded!'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected!'

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            logger.info('Saving upload to %s', file_path)
            file.save(file_path)

            # Detect emotion
            logger.info('Running inference for %s', file.filename)
            emotion, confidence = detect_emotion(file_path)
            logger.info('Inference done in %.2fs', time.time() - started)

            return render_template('index.html', image_path=file_path, emotion=emotion, confidence=confidence)

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, port=port)
