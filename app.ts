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

# Function to predict clause within a section
def predict_clause(input_text, section_name, df, vectorizer, X):
    input_text_preprocessed = preprocess_text(input_text)
    input_text_vectorized = vectorizer.transform([input_text_preprocessed])
    
    section_indices = df[df['section'] == section_name].index
    section_X = X[section_indices]
    section_similarities = cosine_similarity(input_text_vectorized, section_X)
    max_similarity_index = section_indices[section_similarities.argmax()]
    most_similar_clause = df.loc[max_similarity_index]
    
    return most_similar_clause['clause'], most_similar_clause.get('sub_section')

# Function to slice document based on dataset
def slice_document(document_text, section_names, df, vectorizer, X):
    slices = []
    current_section = None
    current_text = []
    paragraphs = document_text.split('\n\n')

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # Check if the paragraph matches any section name
            is_section = False
            for section_name in section_names:
                if re.search(r'\b' + re.escape(section_name.lower()) + r'\b', paragraph.lower()):
                    if current_text and current_section:
                        # Process the collected paragraphs under the current section
                        full_text = " ".join(current_text)
                        clause, sub_section = predict_clause(full_text, current_section, df, vectorizer, X)
                        slices.append((full_text, current_section, clause, sub_section))
                    
                    # Start new section
                    current_section = section_name
                    current_text = []
                    is_section = True
                    break
            
            if not is_section:
                current_text.append(paragraph)
    
    # Add the last collected paragraphs
    if current_text and current_section:
        full_text = " ".join(current_text)
        clause, sub_section = predict_clause(full_text, current_section, df, vectorizer, X)
        slices.append((full_text, current_section, clause, sub_section))

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

    # Get section names from the dataset
    section_names = df['section'].unique()

    # Path to the PDF file
    pdf_path = 'path/to/your/document.pdf'  # Replace with your PDF file path

    # Extract text from the PDF
    document_text = extract_text_from_pdf(pdf_path)
    
    # Slice the document based on the dataset
    slices = slice_document(document_text, section_names, df, vectorizer, X)

    # Print the results
    for sliced_text, section, clause, sub_section in slices:
        print(f"Sliced Text: '{sliced_text}'\nSection: {section}, Clause: {clause}, Sub-section: {sub_section}\n")
