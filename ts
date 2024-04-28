import os
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load dataset using PDF to text extraction function
def extract_text_from_pdf(pdf_path):
    # Your PDF to text extraction function using pdfplumber
    pass

data_path = "path/to/dataset"
categories = os.listdir(data_path)
docs = []
labels = []
for category in categories:
    category_path = os.path.join(data_path, category)
    files = os.listdir(category_path)
    for file in files:
        text = extract_text_from_pdf(os.path.join(category_path, file))
        docs.append(text)
        labels.append(category)

X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.3, random_state=42, shuffle=True)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define preprocessing steps using spaCy
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    # Tokenize text using spaCy
    tokens = nlp(text)
    # Lemmatize tokens
    tokens = [token.lemma_ for token in tokens if not token.is_stop]
    # Rejoin tokens into a string
    text = ' '.join(tokens)
    return text

# list of alpha values
alpha_values = np.linspace(0, 2, num=11)

# Initialize lists to store the accuracy values
train_acc = []
test_acc = []

# Define the pipeline
for alpha in alpha_values:
    nb_pipeline = Pipeline([
        ('preprocess', CountVectorizer(preprocessor=preprocess_text,
                                       ngram_range=(1, 1), max_df=0.8, min_df=2)),
        ('tfidf', TfidfTransformer()),
        ('nb', MultinomialNB(alpha=alpha)),
    ])

    # Fit the model
    print("Fitting")
    nb_pipeline.fit(X_train, y_train)
    print("done")

    # calculate accuracy on training set
    train_predicted = nb_pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predicted)
    train_acc.append(train_accuracy)

    # calculate on test set
    test_predict = nb_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predict)
    test_acc.append(test_accuracy)

# Plot the accuracy values
plt.plot(alpha_values, train_acc, '-o', label='Training Set')
plt.plot(alpha_values, test_acc, '-o', label='Test Set')
plt.xlabel('Alpha Values')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Alpha Values for Multinomial Naive Bayes Model')
plt.legend()
plt.show()

# Predicting Document Types for New Document
new_document_text = "This is a new document text to predict its category."
preprocessed_new_document = preprocess_text(new_document_text)
predicted_category = nb_pipeline.predict([preprocessed_new_document])
print("Predicted category for new document:", predicted_category)
