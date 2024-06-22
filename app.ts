import os
import re
import string
import json
import PyPDF2
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

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

# Find matching segments
def find_matching_segments(pdf_text, dataset_structure, vectorizer):
    preprocessed_text = preprocess_text(pdf_text)
    text_chunks = pdf_text.split('\n')  # Split text into chunks by lines
    
    structure_texts = [f"{entry['section']} {entry['sub_section']}" for entry in dataset_structure]
    structure_vectors = vectorizer.fit_transform(structure_texts)
    
    matches = []
    for chunk in text_chunks:
        chunk_vector = vectorizer.transform([preprocess_text(chunk)])
        similarities = cosine_similarity(chunk_vector, structure_vectors)
        max_sim_idx = similarities.argmax()
        if similarities[0, max_sim_idx] > 0.5:  # Adjust threshold as needed
            matches.append({
                "text": chunk,
                "section": dataset_structure[max_sim_idx]["section"],
                "sub_section": dataset_structure[max_sim_idx]["sub_section"]
            })
        else:
            matches.append({
                "text": chunk,
                "section": None,
                "sub_section": None
            })
    return matches

# Segment text based on identified sections and sub-sections
def segment_text_by_matches(matches):
    segments = {}
    current_section = None
    current_sub_section = None
    
    for match in matches:
        if match["section"]:
            current_section = match["section"]
            current_sub_section = match["sub_section"]
            if current_section not in segments:
                segments[current_section] = {}
            if current_sub_section not in segments[current_section]:
                segments[current_section][current_sub_section] = []
        if current_section and current_sub_section:
            segments[current_section][current_sub_section].append(match["text"])
    
    return segments

# Predict clauses within each segment
def predict_clauses_within_segments(segments, df, vectorizer):
    predictions = []
    for section, sub_sections in segments.items():
        for sub_section, texts in sub_sections.items():
            for text in texts:
                preprocessed_text = preprocess_text(text)
                text_vector = vectorizer.transform([preprocessed_text])
                # Calculate similarity for clauses within the same section
                similarities = {}
                for section_name in df['section'].unique():
                    section_indices = df[df['section'] == section_name].index
                    section_X = X[section_indices]
                    section_similarities = cosine_similarity(text_vector, section_X)
                    max_similarity = section_similarities.max()
                    max_similarity_index = section_indices[section_similarities.argmax()]
                    similarities[section_name] = (max_similarity, max_similarity_index)
                
                # Find the most similar clause across all sections
                most_similar_section = max(similarities, key=lambda x: similarities[x][0])
                most_similar_clause_index = similarities[most_similar_section][1]
                most_similar_clause = df.loc[most_similar_clause_index]

                predictions.append({
                    "section": section,
                    "sub_section": sub_section,
                    "text": text,
                    "predicted_section": most_similar_clause['section'],
                    "predicted_clause": most_similar_clause['clause'],
                    "predicted_sub_section": most_similar_clause['sub_section'] if most_similar_clause['sub_section'] else None
                })
    return predictions

# Convert predictions to JSON
def predictions_to_json(predictions, output_file='predictions.json'):
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

# Process PDF and make predictions
def process_pdf(pdf_path, dataset_structure, df, vectorizer):
    pdf_text = extract_text_from_pdf(pdf_path)
    matches = find_matching_segments(pdf_text, dataset_structure, vectorizer)
    segments = segment_text_by_matches(matches)
    predictions = predict_clauses_within_segments(segments, df, vectorizer)
    return predictions

# Assuming you have already trained your prediction model and TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# Load or define your pre-trained model
# prediction_model = ...

# Load dataset structure
dataset_structure = load_dataset_structure()

# Path to the PDF file
pdf_path = 'path/to/your/pdf'

# Process the PDF and get predictions
predictions = process_pdf(pdf_path, dataset_structure, df, vectorizer)

# Convert predictions to JSON
predictions_to_json(predictions)
