from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS, cross_origin
import cv2
import keras_ocr

app = Flask(__name__)

UPLOAD_FOLDER = 'files'  # Specify the folder where uploaded files will be stored
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}  # Specify allowed file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize the Keras OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

def allowed_file(filename):
    try:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    except Exception as e:
        print("Error in allowed_file:", e)

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify("No file part")

        file = request.files['file']

        if file.filename == '':
            return jsonify("No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)  # Save the uploaded image file

            # Read the uploaded image
            image = cv2.imread(path)

            # Use Keras OCR to recognize text in the image
            predictions = pipeline.recognize([image])

            # Extract the recognized text
            recognized_text = [text for text, _ in predictions[0]]

            print(recognized_text,"recognized_text")

            # Return the recognized text in the response
            return jsonify({"message": "File successfully uploaded.", "recognized_text": recognized_text})

        else:
            return jsonify("Invalid file type. Allowed types: jpg, jpeg, png")

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
