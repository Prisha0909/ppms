import os
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Function to load dataset
def load_dataset(base_dir='dataset/'):
    documents = []
    
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
                        
                        sub_clauses = content.split("###xxx###")
                        for sub_clause in sub_clauses:
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
                                clause_text = sub_clause.strip()
                                if clause_text:
                                    documents.append({
                                        "text": preprocess_text(clause_text),
                                        "section": section_name,
                                        "clause": clause_name,
                                        "sub_section": None
                                    })

    return pd.DataFrame(documents)

# Function to predict section and clause
def predict_section_clause(input_text, vectorizer, df, X):
    input_text_preprocessed = preprocess_text(input_text)
    input_text_vectorized = vectorizer.transform([input_text_preprocessed])
    
    similarities = {}
    for section_name in df['section'].unique():
        section_indices = df[df['section'] == section_name].index
        section_X = X[section_indices]
        section_similarities = cosine_similarity(input_text_vectorized, section_X)
        max_similarity = section_similarities.max()
        max_similarity_index = section_indices[section_similarities.argmax()]
        similarities[section_name] = (max_similarity, max_similarity_index)
    
    most_similar_section = max(similarities, key=lambda x: similarities[x][0])
    most_similar_clause_index = similarities[most_similar_section][1]
    most_similar_clause = df.loc[most_similar_clause_index]
    
    return most_similar_clause['section'], most_similar_clause['clause'], most_similar_clause.get('sub_section')

# Function to slice document based on dataset
def slice_document(document_text, df, vectorizer, X):
    slices = []
    current_section = None
    current_clause = None
    current_text = []

    # Iterate over each paragraph in the document
    for paragraph in document_text.split('\n\n'):
        paragraph = paragraph.strip()
        if paragraph:
            section, clause, sub_section = predict_section_clause(paragraph, vectorizer, df, X)
            if current_section is None:
                current_section = section
                current_clause = clause
                current_sub_section = sub_section
            elif section != current_section or clause != current_clause or sub_section != current_sub_section:
                slices.append((current_text, current_section, current_clause, current_sub_section))
                current_section = section
                current_clause = clause
                current_sub_section = sub_section
                current_text = []
            current_text.append(paragraph)
    
    # Add the last collected text
    if current_text:
        slices.append((current_text, current_section, current_clause, current_sub_section))

    return slices

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text

# Main code execution
if __name__ == "__main__":
    # Load the dataset
    df = load_dataset()

    # Initialize the vectorizer and fit the dataset text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])

    # Path to the PDF file
    pdf_path = 'path/to/your/document.pdf'  # Replace with your PDF file path

    # Extract text from the PDF
    document_text = extract_text_from_pdf(pdf_path)
    
    # Slice the document based on the dataset
    slices = slice_document(document_text, df, vectorizer, X)

    # Print the results
    for sliced_text, section, clause, sub_section in slices:
        print(f"Sliced Text: '{' '.join(sliced_text)}'\nSection: {section}, Clause: {clause}, Sub-section: {sub_section}\n")
