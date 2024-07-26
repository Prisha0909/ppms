import os
import re

# Preprocess the text (e.g., convert to lowercase)
def preprocess_text(text):
    return text.lower()

# Base directory for the clause library
base_dir = "path/to/your/clause_library"

# List to store documents
documents = []

# Load and preprocess the dataset
for superdoc_type in os.listdir(base_dir):
    superdoc_path = os.path.join(base_dir, superdoc_type)
    if os.path.isdir(superdoc_path):
        for major_doc in os.listdir(superdoc_path):
            major_doc_path = os.path.join(superdoc_path, major_doc)
            if os.path.isdir(major_doc_path):
                for section_folder in os.listdir(major_doc_path):
                    section_path = os.path.join(major_doc_path, section_folder)
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

# Now documents contain the preprocessed text and their corresponding labels
print("Number of documents processed:", len(documents))
