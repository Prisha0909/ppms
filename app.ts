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
                        match = re.match(r"###(.+?)###(.+)", sub_clause, re.DOTALL)
                        if match:
                            sub_section_name = match.group(1).strip()
                            clause_text = match.group(2).strip()
                        else:
                            sub_section_name = None
                            clause_text = sub_clause.strip()
                        
                        if clause_text:
                            documents.append({
                                "text": preprocess_text(clause_text),
                                "section": section_name,
                                "clause": clause_name,
                                "sub_section": sub_section_name
                            })

# Convert to DataFrame for easier handling
df = pd.DataFrame(documents)

# Vectorize the dataset using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Get input clause from the user
print("Enter the clause text to compare (press Enter on an empty line to finish input):")
input_lines = []
while True:
    line = input().strip()
    if not line:
        break
    input_lines.append(line)

# Combine the input lines into a single string
input_clause = ' '.join(input_lines)

# Preprocess the input clause text
input_clause_preprocessed = preprocess_text(input_clause)

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

# Output the most similar section name, clause name, and sub-section name (if any)
if most_similar_clause['sub_section']:
    print(most_similar_clause['section'], most_similar_clause['clause'], most_similar_clause['sub_section'])
else:
    print(most_similar_clause['section'], most_similar_clause['clause'])
