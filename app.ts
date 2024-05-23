import os
import re
import string
import pandas as pd
from gensim import corpora
from gensim.models import Word2Vec, KeyedVectors
from gensim.matutils import softcossim
import numpy as np

# Load and preprocess the dataset
documents = []
base_dir = 'dataset/'  # Assuming your dataset is in a folder named 'dataset'

# Traverse through the dataset directory structure
for section_folder in os.listdir(base_dir):
    section_path = os.path.join(base_dir, section_folder)
    if os.path.isdir(section_path):
        for clause_file in os.listdir(section_path):
            clause_path = os.path.join(section_path, clause_file)
            if os.path.isfile(clause_path):
                with open(clause_path, 'r', encoding='utf-8') as file:
                    clause_text = file.read().strip()
                    # Extract the section name from the folder name
                    section_name = section_folder
                    # Remove file extension to get clause name
                    clause_name = os.path.splitext(clause_file)[0]
                    documents.append({"text": clause_text, "label": (section_name, clause_name)})

# Convert to DataFrame for easier handling
df = pd.DataFrame(documents)

# Preprocess the text (lowercase, remove digits and punctuation)
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Apply preprocessing to all clauses in the dataset
df['text'] = df['text'].apply(preprocess_text)

# Load a pre-trained Word2Vec model
model = KeyedVectors.load_word2vec_format('path_to_your_model.bin', binary=True)  # Load your pre-trained Word2Vec model

# Create a dictionary from the dataset
dictionary = corpora.Dictionary([clause.split() for clause in df['text']])

# Create the similarity matrix
similarity_matrix = model.wv.similarity_matrix(dictionary)

def preprocess_text_gensim(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

# Get input clause from the user
print("Enter the clause text to compare (press Enter on an empty line to finish input):")
input_lines = []
while True:
    line = input().strip()  # Read a line of input
    if not line:  # Check if the line is empty (indicating end of input)
        break
    input_lines.append(line)

# Combine the input lines into a single string
input_clause = ' '.join(input_lines)

# Preprocess the input clause text
input_clause_preprocessed = preprocess_text_gensim(input_clause)
input_clause_bow = dictionary.doc2bow(input_clause_preprocessed)

# Calculate soft cosine similarity
similarities = {}
for section_name in df['label'].apply(lambda x: x[0]).unique():
    section_indices = df[df['label'].apply(lambda x: x[0]) == section_name].index
    section_similarities = []
    for index in section_indices:
        clause_bow = dictionary.doc2bow(preprocess_text_gensim(df['text'][index]))
        sim_score = softcossim(input_clause_bow, clause_bow, similarity_matrix)
        section_similarities.append((sim_score, index))
    
    max_similarity, max_similarity_index = max(section_similarities, key=lambda x: x[0])
    similarities[section_name] = (max_similarity, max_similarity_index)

# Find the most similar clause across all sections
most_similar_section = max(similarities, key=lambda x: similarities[x][0])
most_similar_clause_index = similarities[most_similar_section][1]
most_similar_clause = df.loc[most_similar_clause_index]

# Output the most similar section name and clause name
print(most_similar_clause['label'][0], most_similar_clause['label'][1])
