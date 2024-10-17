from flask import Flask, request, jsonify, send_file
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
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
