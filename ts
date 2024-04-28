import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pdfplumber

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf = pdfplumber.open(f)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Load and preprocess the dataset
def load_dataset(data_path):
    docs = []
    labels = []
    categories = os.listdir(data_path)
    for category in categories:
        category_path = os.path.join(data_path, category)
        files = os.listdir(category_path)
        for file in files:
            text = extract_text_from_pdf(os.path.join(category_path, file))
            docs.append(text)
            labels.append(category)
    return docs, labels

data_path = "path/to/dataset"
X_text, y = load_dataset(data_path)

# Split the dataset into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y)))

# Tokenize and encode the text data
X_train_encoded = tokenizer(X_train_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
X_test_encoded = tokenizer(X_test_text, padding=True, truncation=True, max_length=512, return_tensors='pt')

# Convert labels to numerical indices
label_map = {label: i for i, label in enumerate(set(y))}
y_train_encoded = [label_map[label] for label in y_train]
y_test_encoded = [label_map[label] for label in y_test]

# Convert data to PyTorch tensors
X_train_tensors = TensorDataset(X_train_encoded['input_ids'], X_train_encoded['attention_mask'], torch.tensor(y_train_encoded))
X_test_tensors = TensorDataset(X_test_encoded['input_ids'], X_test_encoded['attention_mask'], torch.tensor(y_test_encoded))

# Create DataLoader for training and testing sets
train_loader = DataLoader(X_train_tensors, batch_size=4, shuffle=True)
test_loader = DataLoader(X_test_tensors, batch_size=4, shuffle=False)

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for epoch in range(3):  # Adjust number of epochs as needed
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

# Function to predict document type from a given PDF
def predict_document_type(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    encoded_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids, attention_mask = encoded_text['input_ids'].to(device), encoded_text['attention_mask'].to(device)
    output = model(input_ids, attention_mask=attention_mask)
    predicted_label_idx = torch.argmax(output.logits, dim=1).item()
    predicted_label = [label for label, idx in label_map.items() if idx == predicted_label_idx][0]
    return predicted_label

# Example usage
pdf_path = "path/to/your/pdf/file.pdf"
predicted_type = predict_document_type(pdf_path)
print("Predicted document type:", predicted_type)
