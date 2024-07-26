import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pdfplumber

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

# Function to predict the category of a PDF file
def predict_category_with_confidence(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_text(text)
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequences)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = prediction[0][predicted_class_idx]
    return predicted_class, confidence

# Load dataset
dataset_dir = "path/to/your/document_library"
superdocument_types = os.listdir(dataset_dir)

X = []  # List to store preprocessed text data
y = []  # List to store corresponding labels

for superdoc_type in superdocument_types:
    superdoc_path = os.path.join(dataset_dir, superdoc_type)
    if os.path.isdir(superdoc_path):
        major_documents = os.listdir(superdoc_path)
        for major_doc in major_documents:
            major_doc_path = os.path.join(superdoc_path, major_doc)
            if os.path.isdir(major_doc_path):
                document_types = os.listdir(major_doc_path)
                for doc_type in document_types:
                    doc_type_path = os.path.join(major_doc_path, doc_type)
                    if os.path.isdir(doc_type_path):
                        files = os.listdir(doc_type_path)
                        for file in files:
                            file_path = os.path.join(doc_type_path, file)
                            if file.endswith(".pdf"):
                                text = extract_text_from_pdf(file_path)
                                processed_text = preprocess_text(text)
                                X.append(processed_text)
                                y.append(doc_type)  # Assuming folder name represents the category

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
max_length = 1000  # Adjust as needed

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

# Example usage of prediction with confidence
pdf_file_path = "path/to/your/pdf/file.pdf"
predicted_category, confidence = predict_category_with_confidence(pdf_file_path)
print("Predicted Category:", predicted_category)
print("Confidence:", confidence)
