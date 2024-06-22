import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import json

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# Function to predict clauses within a segment
def predict_clauses_within_segment(segment_content, df, vectorizer):
    # Preprocess segment content
    preprocessed_segment = preprocess_text(segment_content)
    
    # Vectorize the segment content
    segment_vector = vectorizer.transform([preprocessed_segment])
    
    # Calculate similarity for clauses within the same section
    similarities = {}
    for index, row in df.iterrows():
        section_name = row['section']
        clause_name = row['clause']
        sub_section_name = row['sub_section']
        clause_text = row['text']
        
        clause_vector = vectorizer.transform([preprocess_text(clause_text)])
        
        similarity = cosine_similarity(segment_vector, clause_vector)
        similarities[(section_name, clause_name, sub_section_name)] = similarity[0][0]
    
    # Find top matches
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]  # Adjust the number of top matches as needed
    
    # Prepare clauses with their similarity scores
    clauses = []
    for (section_name, clause_name, sub_section_name), similarity in top_matches:
        clauses.append({
            "section_name": section_name,
            "sub_section_name": sub_section_name,
            "clause_name": clause_name,
            "similarity_score": similarity
        })
    
    return clauses

# Function to process each PDF
def process_pdf(pdf_path, df, vectorizer):
    with pdfplumber.open(pdf_path) as pdf:
        predictions = {}
        
        for page in pdf.pages:
            text = page.extract_text()
            
            # Example splitting logic (adjust as per your actual format)
            segments = re.split(r"###(.+?)###", text)
            
            for i in range(1, len(segments), 2):
                section_name = segments[i].strip()
                segment_content = segments[i + 1].strip() if i + 1 < len(segments) else ""
                
                # Predict clauses within this segment
                if segment_content:
                    clauses = predict_clauses_within_segment(segment_content, df, vectorizer)
                    
                    # Accumulate predictions by section_name
                    if section_name not in predictions:
                        predictions[section_name] = {}
                    
                    for clause in clauses:
                        sub_section_name = clause['sub_section_name']
                        if sub_section_name not in predictions[section_name]:
                            predictions[section_name][sub_section_name] = []
                        predictions[section_name][sub_section_name].append({
                            "clause_name": clause['clause_name'],
                            "similarity_score": clause['similarity_score']
                        })
        
        return predictions

# Example usage
if __name__ == "__main__":
    base_dir = 'path_to_your_dataset_directory'  # Replace with the directory path where your clauses are stored
    
    # Example: Load your dataset from directory structure
    documents = []
    
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
    
    # Initialize your vectorizer
    vectorizer = TfidfVectorizer()  # You can configure the vectorizer with parameters as needed
    
    # Example: Path to your PDF file
    pdf_path = 'path_to_your_pdf.pdf'  # Replace with the actual path to your PDF file
    
    # Process the PDF
    predictions = process_pdf(pdf_path, df, vectorizer)
    
    # Convert predictions to JSON for output
    predictions_json = json.dumps(predictions, indent=4)
    print(predictions_json)
