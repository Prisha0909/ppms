import os
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

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

def parse_document(document, vectorizer, df, X):
    sections = re.split(r'\n\n+', document)  # Split sections by double newlines
    results = []
    for section in sections:
        sub_clauses = re.split(r'\n+', section)  # Split clauses by single newlines
        for sub_clause in sub_clauses:
            if sub_clause.strip():
                result = predict_section_clause(sub_clause, vectorizer, df, X)
                results.append((sub_clause.strip(), result))
    return results

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

if __name__ == "__main__":
    df = load_dataset()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])

    pdf_path = 'path/to/your/document.pdf'  # Replace with your PDF file path
    document_text = extract_text_from_pdf(pdf_path)
    
    parsed_results = parse_document(document_text, vectorizer, df, X)

    for sub_clause, (section, clause, sub_section) in parsed_results:
        if sub_section:
            print(f"Sub-clause: '{sub_clause}'\nPredicted Section: {section}, Clause: {clause}, Sub-section: {sub_section}\n")
        else:
            print(f"Sub-clause: '{sub_clause}'\nPredicted Section: {section}, Clause: {clause}\n")
