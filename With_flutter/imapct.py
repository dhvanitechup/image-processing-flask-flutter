# UPLOAD_FOLDER = 'files'  # Specify the folder where uploaded files will be stored
# ALLOWED_EXTENSIONS = {'jpg','png','jpeg'}  # Specify allowed file extensions

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# def allowed_file(filename):
#     try:
#         return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#     except Exception as e:
#         print("error in allowed_file", e)

# def unzip(path, pt):
#     try:
#         #print (path)
#         with zipfile.ZipFile(path,"r") as zip_ref:
#             zip_ref.extractall(pt)
#             return 1
#     except Exception as e:
#         print("error in unzip", e)


# CORS(app, resources={r"/*": {"origins": "*"}})

# @app.route('/eob_load', methods=['POST'])
# @cross_origin()
# def upload_file():
#     try:
#         if 'file' not in request.files:
#             return jsonify("No file part")

#         file = request.files['file']

#         if file.filename == '':
#             return jsonify("No selected file")

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(path)  # Save the uploaded file

#             pt = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0])

#             a = unzip(path, pt)











from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS, cross_origin

app = Flask(__name__)

UPLOAD_FOLDER = 'files'  # Specify the folder where uploaded files will be stored
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}  # Specify allowed file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app, resources={r"/*": {"origins": "*"}})

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

            return jsonify("File successfully uploaded.")

        else:
            return jsonify("Invalid file type. Allowed types: jpg, jpeg, png")

    except Exception as e:
        return jsonify("Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)



