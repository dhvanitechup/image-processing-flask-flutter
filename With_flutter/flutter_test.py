# from flask import Flask, request, send_file, jsonify
# import os
# from io import BytesIO

# app = Flask(__name__)

# # Set the folder to upload and download files
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     # Get the binary data from the request body
#     file_data = request.get_data()

#     if not file_data:
#         return jsonify({'error': 'No file provided'}), 400

#     # Save the binary data to a file in the specified folder
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
#     with open(file_path, 'wb') as f:
#         f.write(file_data)

#     return jsonify({'message': 'File uploaded successfully'}), 200

# @app.route('/download')
# def download_file():
#     # Set the file path for download
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')

#     # Send the file as a response
#     return send_file(file_path, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)





# from flask import Flask, request, jsonify
# import os

# app = Flask(__name__)

# # Specify the folder where uploaded files will be stored
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     # Get the JSON data from the request
#     data = request.get_json()

#     # Check if the required fields are present in the JSON data
#     if 'filename' not in data or 'file' not in data:
#         return jsonify({'error': 'Invalid JSON format'}), 400

#     # Extract filename and file from JSON data
#     filename = data['filename']
#     file_content = data['file']

#     # Decode base64 file content (assuming it's base64 encoded)
#     file_data = file_content.split(',')[1].encode('utf-8')
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#     # Save the file to the specified folder
#     with open(file_path, 'wb') as f:
#         f.write(file_data)

#     return jsonify({'message': 'File successfully uploaded to {} folder as {}'.format(UPLOAD_FOLDER, filename)})

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Specify the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the 'file' key is in the request.files dictionary
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser may submit an empty part without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get the filename from the form (you can modify this as needed)
    filename = request.form.get('filename', 'uploaded_file')

    # Save the file to the specified folder
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return jsonify({'message': 'File successfully uploaded to {} folder as {}'.format(UPLOAD_FOLDER, filename)})

if __name__ == '__main__':
    app.run(debug=True)

