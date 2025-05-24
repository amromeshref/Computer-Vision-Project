from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from utils import getSimilarImages
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_similar', methods=['POST'])
def find_similar():
    file = request.files['image']
    type = request.form.get('type')  # e.g., "jeans"
    
    # Save image
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Read and process image
    image = cv2.imread(path)
    similar = getSimilarImages(image, type)

    # Prepare response
    results = []
    for sim_img, info in similar:
        _, img_encoded = cv2.imencode('.jpg', sim_img)
        img_bytes = img_encoded.tobytes()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        results.append({'info': info, 'image': img_b64})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
