from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import os
from classify_model import classify_document
from clause_model import predict_clauses

app = Flask(__name__)
CORS(app)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract text from PDF
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()

    # Classify the document
    doc_type = classify_document(file_path)

    # Predict clauses
    clauses = predict_clauses(text)

    return jsonify({'doc_type': doc_type, 'text': text, 'clauses': clauses})

if __name__ == '__main__':
    app.run(debug=True)
