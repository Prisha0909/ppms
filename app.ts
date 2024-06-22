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

                predictions.append({
                    "section": section,
                    "sub_section": sub_section,
                    "text": text,
                    "predicted_section": most_similar_clause['section'],
                    "predicted_clause": most_similar_clause['clause'],
                    "predicted_sub_section": most_similar_clause['sub_section'] if most_similar_clause['sub_section'] else None
                })
    return predictions
