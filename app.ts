import pdfplumber
import os
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Function to extract text segments from a PDF
def extract_segments_from_pdf(pdf_path):
    segments = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Implement your logic to segment text into sections and sub-sections
            # Example logic: split by section headings, etc.
            # For demonstration, assume a simple split by lines
            lines = text.split('\n')
            section = None
            for line in lines:
                if line.startswith('###'):
                    section = line.strip('###').strip()
                elif section:
                    segments.append({
                        "text": line.strip(),
                        "section": section
                    })
    return segments

# Function to predict clauses within segments
def predict_clauses_within_segments(segments, df, vectorizer):
    predictions = []
    for segment in segments:
        text = segment['text']
        section = segment['section']
        
        preprocessed_text = preprocess_text(text)
        text_vector = vectorizer.transform([preprocessed_text])
        
        # Calculate similarity for clauses within the same section
        similarities = {}
        for section_name in df['section'].unique():
            section_indices = df[df['section'] == section_name].index
            section_X = X[section_indices]
            
            # Ensure section_X matches the vectorizer's vocabulary
            section_X_transformed = vectorizer.transform(df.loc[section_indices, 'text'])
            
            section_similarities = cosine_similarity(text_vector, section_X_transformed)
            max_similarity = section_similarities.max()
            max_similarity_index = section_indices[section_similarities.argmax()]
            similarities[section_name] = (max_similarity, max_similarity_index)
        
        # Find the most similar clause across all sections
        most_similar_section = max(similarities, key=lambda x: similarities[x][0])
        most_similar_clause_index = similarities[most_similar_section][1]
        most_similar_clause = df.loc[most_similar_clause_index]
        
        # Prepare the prediction result
        prediction = {
            "text": text,
            "predicted_section": most_similar_clause['section'],
            "predicted_clause": most_similar_clause['clause'],
            "predicted_sub_section": most_similar_clause['sub_section'] if most_similar_clause['sub_section'] else None
        }
        predictions.append(prediction)
    
    return predictions

# Main function to process PDF and make predictions
def process_pdf(pdf_path, df, vectorizer):
    segments = extract_segments_from_pdf(pdf_path)
    predictions = predict_clauses_within_segments(segments, df, vectorizer)
    return predictions

# Example usage
if __name__ == "__main__":
    # Load and preprocess the dataset
    documents = []
    base_dir = 'dataset/'

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

    # Example PDF processing
    pdf_path = 'example.pdf'
    predictions = process_pdf(pdf_path, df, vectorizer)

    # Print or save predictions as needed
    for prediction in predictions:
        result = {
            "text": prediction['text'],
            "predicted_section": prediction['predicted_section'],
            "predicted_clause": prediction['predicted_clause'],
            "predicted_sub_section": prediction['predicted_sub_section']
        }
        print(json.dumps(result, indent=4))
