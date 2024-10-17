from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS, cross_origin
import cv2
import keras_ocr
import numpy as np

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



def find_and_draw_rectangle(image, text_to_find):
    try:
        # Use Keras OCR to recognize text in the image
        predictions = pipeline.recognize([image])

        img_with_rectangle = image.copy()
        found = False

        for text, box in predictions[0]:
            if text == text_to_find:
                found = True

                # Convert box coordinates to integers
                box = [(int(coord[0]), int(coord[1])) for coord in box]

                # Draw a white rectangle around the specified text
                img_with_rectangle = cv2.rectangle(img_with_rectangle, box[0], box[2], (255, 0, 0), 2)

                # Create a new contour using the bounding box coordinates
                new_contour = [
                    (box[0][0], box[0][1]),
                    (box[2][0], box[0][1]),
                    (box[2][0], box[2][1]),
                    (box[0][0], box[2][1]),
                ]

                new_contour = [tuple(map(int, point)) for point in new_contour]

                # Convert the new_contour to a NumPy array
                new_contour_np = np.array(new_contour)

                # Convert the bounding box region to grayscale
                gray = cv2.cvtColor(img_with_rectangle, cv2.COLOR_BGR2GRAY)

                # Now convert the grayscale image to a binary image
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Now detect the contours in the bounding box region
                contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

                for contour_num, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)

                    new_contour_x = new_contour[0][0] - 3
                    new_contour_y = new_contour[0][1] - 3
                    new_contour_x_01 = new_contour[0][0] + 3
                    new_contour_y_02 = new_contour[0][1] + 3

                    if new_contour_x <= x <= new_contour_x_01 and new_contour_y <= y <= new_contour_y_02:
                        parent_number = hierarchy[0][contour_num][3]
                        
                        img_with_rectangle = cv2.drawContours(img_with_rectangle, [contour], -1, (0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

                        if parent_number > -1:
                            img_with_rectangle = cv2.drawContours(img_with_rectangle, contours[parent_number], -1, (0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

                            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.jpg')
                            cv2.imwrite(result_image_path, img_with_rectangle)

                            return result_image_path

        if not found:
            print(f"Text '{text_to_find}' not found in the image.")

        return None

    except Exception as e:
        print("Error in find_and_draw_rectangle:", e)
        return None



@app.route('/search', methods=['POST'])
@cross_origin()
def search_image():
    try:
        # Get image name and text from Postman request
        image_name = request.form.get('image_name')
        text_to_find = request.form.get('text_to_find')

        # Load the image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        image = cv2.imread(image_path)

        # Find and draw rectangle and contours for the specified text
        result_image_path = find_and_draw_rectangle(image, text_to_find)

        if result_image_path:
            return jsonify({"message": "Text found and contours drawn successfully.", "result_image_path": result_image_path})
        else:
            return jsonify({"message": f"Text '{text_to_find}' not found in the image.", "result_image_path": None})

    except Exception as e:
        return jsonify({"error": str(e)})


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


