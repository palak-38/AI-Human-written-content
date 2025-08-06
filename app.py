from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from docx import Document
import torch
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import json
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Load your XGBoost classifier model
with open('xgb_classifier.json', 'r') as f:
    xgb_model = xgb.Booster(model_file=f.read())

# Allowed extensions for uploads
ALLOWED_EXTENSIONS = {'txt', 'docx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_docx(file_path):
    """Extract text from .docx file"""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def preprocess_text(input_text):
    """Tokenize and encode text using BERT"""
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the [CLS] token output (first token embedding)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
    return cls_embedding


@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    input_text = data.get('text', '')

    if not input_text.strip():
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess and encode text
    cls_embedding = preprocess_text(input_text)

    # Reshape for XGBoost input
    cls_embedding = np.expand_dims(cls_embedding, axis=0)

    # Classify the text
    prediction = xgb_model.predict(xgb.DMatrix(cls_embedding))[0]

    return jsonify({'prediction': int(prediction)})


@app.route('/upload-files', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    results = {}

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Extract text
            if filename.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

            # Preprocess and encode text
            cls_embedding = preprocess_text(text)

            # Reshape for XGBoost input
            cls_embedding = np.expand_dims(cls_embedding, axis=0)

            # Classify the text
            prediction = xgb_model.predict(xgb.DMatrix(cls_embedding))[0]
            results[filename] = int(prediction)

            # Clean up the file
            os.remove(file_path)

    return jsonify(results)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
