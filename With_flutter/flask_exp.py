
from flask import Flask, render_template, request, redirect, url_for
import cv2
import keras_ocr
import numpy as np
from werkzeug.utils import secure_filename
import os
import re

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the pipeline for text recognition
text_finder = None
original_image = None  # Initialize original_image variable

class TextFinder:
    def __init__(self, image_path):
        # Initialize the pipeline for text recognition
        self.pipeline = keras_ocr.pipeline.Pipeline()
        # Read the image
        self.image = keras_ocr.tools.read(image_path)
        
                # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale image to three channels
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # # Read the EDSR model and apply super-resolution
        # upsampled_image = cv2.dnn_superres.DnnSuperResImpl_create()
        # upsampled_image.readModel("EDSR_x4.pb")  # Download the model from https://github.com/opencv/opencv/wiki/Intel%27s-Image-Processor
        # upsampled_image.setModel("edsr", 4)  # 4x upscaling factor

        # Upsample the color image (not the grayscale one)
        print("super")
        self.super_res_image = color_image.copy()
        # enhance_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enhance_image1.jpg')
        # self.enhance_image_write = cv2.imwrite(enhance_image_path, self.super_res_image)
        print("1")

    # def enhance_image(self,image_path):
    #     # Load the image
    #     # input_image = cv2.imread(image_path)

    #     return super_res_image

    def find_and_draw_rectangle(self, text_to_find, predictions):
        img_with_rectangle = self.super_res_image.copy()
        found = False
        contours_for_hierarchy = []

        for text, box in predictions:
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

                # print(box[0][0],"box[0][0]")
                # print(box[0][1],"box[0][1]")
                # print(box[2][0],"box[2][0]")
                # print(box[2][1],"box[2][1]")
                new_contour = [tuple(map(int, point)) for point in new_contour]

                # Convert the new_contour to a NumPy array
                new_contour_np = np.array(new_contour)

                # Convert the bounding box region to grayscale
                gray = cv2.cvtColor(img_with_rectangle, cv2.COLOR_BGR2GRAY)

                # Now convert the grayscale image to a binary image
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Now detect the contours in the bounding box region
                contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

                # Display hierarchy
                hierarchy = np.vstack(hierarchy)
                print("here hierarchy")
                for contour_num, contour in enumerate(contours):
                    
                    # Get the bounding rectangle for the contour
                    x, y, w, h = cv2.boundingRect(contour)

                    new_contour_x = new_contour[0][0] - 3
                    new_contour_y = new_contour[0][1] - 3
                    new_contour_x_01 = new_contour[0][0] + 3
                    new_contour_y_02 = new_contour[0][1] + 3
                    
                    if new_contour_x <= x <= new_contour_x_01 and new_contour_y <= y <= new_contour_y_02:

                        print("enter into if loop after matching")
                        parent_number = hierarchy[contour_num][3]
                        print(type(parent_number),"type_parent_number")
                        # Draw the contour on the image
                        img_with_rectangle = cv2.drawContours(img_with_rectangle, [contour], -1, (0, 255, 0),
                                                            thickness=5, lineType=cv2.LINE_AA)
                        # cv2.putText(img_with_rectangle, f'Contour {contour_num}', (x, y - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                        contours_for_hierarchy.append(contour_num)
                        print("---------------------")
                        print(parent_number,"parent_number")
                        print("---------------")

                        if parent_number > -1:
                            img_with_rectangle = cv2.drawContours(img_with_rectangle, contours[parent_number], -1,
                                                            (0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

                            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image1.jpg')
                            img_with_rectangle = cv2.imwrite(result_image_path, img_with_rectangle)
                            print("find text and area")
                            return "Find text and area succefully"    
                            # return img_with_rectangle
                                                 

                        else:
                            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image1.jpg')
                            img_with_rectangle = cv2.imwrite(result_image_path, img_with_rectangle)
                            print("Area not found")
                            return "Area not found"
                            # return img_with_rectangle


        if not found:
            print(f"Text '{text_to_find}' not found in the image.")

        # return contours_for_hierarchy

    def recognize_text(self):

        # Generate text predictions from the z
        prediction_groups = self.pipeline.recognize([self.super_res_image])

        # Print the recognized text
        predicted_image = prediction_groups[0]
        numeric_values = []
        for text, _ in predicted_image:
            # Use regular expression to find numeric values
            numeric_match = re.findall(r'\d+', text)
            numeric_values.extend(numeric_match)
        print("prd--",predicted_image)
        print("num",numeric_values)
        return predicted_image,numeric_values

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    global text_finder, original_image

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                print("2")
                text_finder = TextFinder(file_path)
                print("3")
                predicted_image, numeric_values = text_finder.recognize_text()
                
                print("4")
                original_image = filename  # Set the original_image variable
                # original_image = "enhance_image1.jpg"  # Set the original_image variable
                result_image = None  # Clear the result_image variable

                return render_template('index.html', original_image=original_image, result_image=result_image,numeric_values=numeric_values)

        elif 'text' in request.form and text_finder:
            text_to_find = request.form['text']
            predicted_image, numeric_values   = text_finder.recognize_text()
            contours_result = text_finder.find_and_draw_rectangle(text_to_find,predicted_image)

            # print(contours_result,"contours_result")
            # result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image_test.jpg')
            # img_with_rectangle_test = cv2.imwrite(result_image_path, contours_result)

            result_image = 'result_image1.jpg'
            # original_image = original_image  # Keep the original_image variable unchanged

            return render_template('index.html', original_image=original_image, result_image=result_image,
                                   contours_result=contours_result,numeric_values=numeric_values)

    return render_template('index.html')  # Ensure original_image is passed to the template




@app.route('/next', methods=['GET'])
def next_page():
    return render_template('createpage.html')

@app.route('/search', methods=['GET'])
def search_image():
    image_name = request.args.get('image_name')
    # Perform search logic based on the entered image_name
    # You can customize this logic based on your requirements

    # # For now, let's redirect to the original page with a message
    # return redirect(url_for('index', message=f'Searching for image: {image_name}'))
    #  image_name = request.args.get('image_name')

    # You can perform additional search logic based on the entered image_name
    # For now, let's redirect to the original page with the searched image name
    return render_template('createpage.html', searched_image_name=image_name)

if __name__ == '__main__':
    app.run(debug=True)






