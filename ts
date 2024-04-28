import os
import pdfplumber
import gensim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf = pdfplumber.open(f)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Function to extract structural features from PDF text
def extract_structural_features(text):
    num_pages = text.count('\f')
    num_chars = len(text)
    num_words = len(text.split())
    num_paragraphs = len(text.split('\n\n'))
    return [num_pages, num_chars, num_words, num_paragraphs]

# Function to load and preprocess the document dataset
def load_dataset(data_path, test_size=0.2, random_state=42):
    texts = []
    structural_features = []
    labels = []

    categories = os.listdir(data_path)
    for category in categories:
        category_path = os.path.join(data_path, category)
        files = os.listdir(category_path)
        for file in files:
            pdf_path = os.path.join(category_path, file)
            text = extract_text_from_pdf(pdf_path)
            if text:
                features = extract_structural_features(text)
                texts.append(gensim.utils.simple_preprocess(text))  # Tokenize text for Doc2Vec
                structural_features.append(features)
                labels.append(category)

    # Split the dataset into training and testing sets
    X_text_train, X_text_test, X_structural_train, X_structural_test, y_train, y_test = train_test_split(
        texts, structural_features, labels, test_size=test_size, random_state=random_state)

    return X_text_train, X_text_test, X_structural_train, X_structural_test, y_train, y_test

# Load and preprocess the dataset
data_path = "path/to/dataset"
X_text_train, X_text_test, X_structural_train, X_structural_test, y_train, y_test = load_dataset(data_path)

# Train Doc2Vec model
documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(X_text_train)]
doc2vec_model = gensim.models.Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)

# Infer document vectors for training and testing data
X_doc2vec_train = np.array([doc2vec_model.infer_vector(doc) for doc in X_text_train])
X_doc2vec_test = np.array([doc2vec_model.infer_vector(doc) for doc in X_text_test])

# Combine Doc2Vec embeddings with structural features
X_train_combined = np.array([doc_vec + struct_feat for doc_vec, struct_feat in zip(X_doc2vec_train, X_structural_train)])
X_test_combined = np.array([doc_vec + struct_feat for doc_vec, struct_feat in zip(X_doc2vec_test, X_structural_test)])


# Scale combined features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train_scaled, y_train)

# Predict labels for test set
y_pred = clf.predict(X_test_scaled)

# Evaluate classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to preprocess and extract features from a new document
def preprocess_new_document(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if text:
        text_tokens = gensim.utils.simple_preprocess(text)
        structural_features = extract_structural_features(text)
        return text_tokens, structural_features
    else:
        return None, None

# Function to predict the type of a new document
def predict_document_type(pdf_path):
    text_tokens, structural_features = preprocess_new_document(pdf_path)
    if text_tokens and structural_features:
        doc_vector = doc2vec_model.infer_vector(text_tokens)
        combined_features = doc_vector + structural_features
        scaled_features = scaler.transform([combined_features])
        predicted_type = clf.predict(scaled_features)[0]
        return predicted_type
    else:
        return None

# Example usage:
new_document_path = "path/to/new/document.pdf"
predicted_type = predict_document_type(new_document_path)
if predicted_type:
    print("Predicted document type:", predicted_type)
else:
    print("Error: Unable to predict document type.")
