# classify_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pdfplumber

# Define global variables for tokenizer and label encoder
tokenizer = None
label_encoder = None
max_length = 1000

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to preprocess text data
def preprocess_text(text):
    processed_text = text.lower()  # Example: Convert text to lowercase
    return processed_text

# Function to train the model (call this once during initialization)
def train_model():
    global tokenizer, label_encoder

    dataset_dir = "path/to/your/Documents/folder"
    folders = os.listdir(dataset_dir)

    X = []  # List to store preprocessed text data
    y = []  # List to store corresponding labels

    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                if file.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                    processed_text = preprocess_text(text)
                    X.append(processed_text)
                    y.append(folder)  # Assuming folder name represents the category

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenize and pad sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length, padding='post')

    # Define CNN model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    num_classes = len(np.unique(y))  # Number of output classes
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    # Save the trained model
    model.save('document_classifier.h5')

    # Save the tokenizer and label encoder
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

# Function to predict the category of a PDF file
def classify_document(pdf_path):
    model = tf.keras.models.load_model('document_classifier.h5')

    text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_text(text)
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequences)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class

-----------------------------------------------------------------------------------------
    # clause_model.py

import os
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the dataset
documents = []
base_dir = 'dataset/'

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Traverse through the dataset directory structure
for section_folder in os.listdir(base_dir):
    section_path = os.path.join(base_dir, section_folder)
    if os.path.isdir(section_path):
        for clause_file in os.listdir(section_path):
            clause_path = os.path.join(section_path, clause_file)
            if os.path.isfile(clause_path):
                with open(clause_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    section_name = section_folder
                    clause_name = os.path.splitext(clause_file)[0]
                    
                    # Split content by the delimiter ###xxx###
                    sub_clauses = content.split("###xxx###")
                    for sub_clause in sub_clauses:
                        # Find all matches of the pattern ###Sub-section name###Sub-clause text
                        matches = re.findall(r"###(.+?)###(.+)", sub_clause, re.DOTALL)
                        if matches:
                            for match in matches:
                                sub_section_name = match[0].strip()
                                clause_text = match[1].strip()
                                if clause_text:
                                    documents.append({
                                        "text": preprocess_text(clause_text),
                                        "section": section_name,
                                        "clause": clause_name,
                                        "sub_section": sub_section_name
                                    })
                        else:
                            # No sub-section name, just use the entire sub_clause text
                            clause_text = sub_clause.strip()
                            if clause_text:
                                documents.append({
                                    "text": preprocess_text(clause_text),
                                    "section": section_name,
                                    "clause": clause_name,
                                    "sub_section": None
                                })

# Convert to DataFrame for easier handling
df = pd.DataFrame(documents)

# Vectorize the dataset using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Function to predict clauses for given text
def predict_clauses(text):
    paragraphs = text.split('\n\n')  # Splitting text into paragraphs
    clauses = []
    for para in paragraphs:
        if not para.strip():
            continue  # Skip empty paragraphs
        # Preprocess the input clause text
        input_clause_preprocessed = preprocess_text(para)
        # Vectorize the input clause
        input_clause_vectorized = vectorizer.transform([input_clause_preprocessed])
        # Calculate similarity for clauses within the same section
        similarities = {}
        for section_name in df['section'].unique():
            section_indices = df[df['section'] == section_name].index
            section_X = X[section_indices]
            section_similarities = cosine_similarity(input_clause_vectorized, section_X)
            max_similarity = section_similarities.max()
            max_similarity_index = section_indices[section_similarities.argmax()]
            similarities[section_name] = (max_similarity, max_similarity_index)
        # Find the most similar clause across all sections
        most_similar_section = max(similarities, key=lambda x: similarities[x][0])
        most_similar_clause_index = similarities[most_similar_section][1]
        most_similar_clause = df.loc[most_similar_clause_index]
        # Append the result
        clauses.append({
            "text": para,
            "section": most_similar_clause['section'],
            "clause": most_similar_clause['clause'],
            "sub_section": most_similar_clause['sub_section'] if most_similar_clause['sub_section'] else None
        })
    return clauses

------------------------------------------------------------
    # app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
from classify_model import classify_document, train_model, extract_text_from_pdf
from clause_model import predict_clauses

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the models by training
train_model()

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Classify the document
    doc_type = classify_document(file_path)

    # Extract text from PDF
    text = extract_text_from_pdf(file_path)

    # Predict clauses
    clauses = predict_clauses(text)

    return jsonify({'doc_type': doc_type, 'text': text, 'clauses': clauses})

if __name__ == '__main__':
    app.run(debug=True)
------------------------------------------
    curl -X POST http://127.0.0.1:5000/upload-pdf -F "file=@/path/to/your/pdf"
