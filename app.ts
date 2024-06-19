import pdfplumber
import os
import re
import string
import json

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Load known sections and subsections from the dataset
base_dir = 'dataset/'
known_sections = {}

for section_folder in os.listdir(base_dir):
    section_path = os.path.join(base_dir, section_folder)
    if os.path.isdir(section_path):
        known_sections[section_folder] = []
        for clause_file in os.listdir(section_path):
            clause_name = os.path.splitext(clause_file)[0]
            known_sections[section_folder].append(clause_name)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    document_text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            document_text += page.extract_text() + '\n'
    return document_text

# Function to split document into sections and subsections
def split_document(document_text, known_sections):
    sections = []
    section_patterns = [re.escape(sec) for sec in known_sections.keys()]
    pattern = re.compile(r'(' + '|'.join(section_patterns) + r')', re.IGNORECASE)
    
    split_points = [(m.start(), m.group()) for m in pattern.finditer(document_text)]
    split_points.append((len(document_text), ''))

    for i in range(len(split_points) - 1):
        start, section_name = split_points[i]
        end, _ = split_points[i + 1]
        section_text = document_text[start:end].strip()
        
        if section_text:
            subsections = split_into_subsections(section_text, section_name)
            sections.append({
                "section_name": section_name,
                "subsections": subsections
            })
    
    return sections

# Function to split section into subsections
def split_into_subsections(section_text, section_name):
    subsections = []
    subsection_patterns = [re.escape(sub) for sub in known_sections[section_name]]
    pattern = re.compile(r'(' + '|'.join(subsection_patterns) + r')', re.IGNORECASE)

    split_points = [(m.start(), m.group()) for m in pattern.finditer(section_text)]
    split_points.append((len(section_text), ''))

    for i in range(len(split_points) - 1):
        start, subsection_name = split_points[i]
        end, _ = split_points[i + 1]
        subsection_text = section_text[start:end].strip()
        
        if subsection_text:
            subsections.append({
                "subsection_name": subsection_name,
                "subsection_content": subsection_text
            })
    
    return subsections

# Function to process uploaded document
def process_uploaded_document(file_path):
    document_text = extract_text_from_pdf(file_path)
    sections = split_document(document_text, known_sections)
    
    output = ""
    for section in sections:
        output += f"Section name: {section['section_name']}\n"
        for idx, subsection in enumerate(section['subsections'], start=1):
            output += f"Subsection{idx} name: {subsection['subsection_name']}\n"
            output += f"Subsection{idx} content: {subsection['subsection_content']}\n"
    
    return output

# Example usage:
file_path = 'uploaded_document.pdf'  # Path to the uploaded document
result_output = process_uploaded_document(file_path)
print(result_output)
# Function to split document into sections and subsections
def split_document(document_text, known_sections):
    sections = []
    section_patterns = [re.escape(sec) for sec in known_sections.keys()]
    pattern = re.compile(r'(' + '|'.join(section_patterns) + r')', re.IGNORECASE)
    
    split_points = [(m.start(), m.group()) for m in pattern.finditer(document_text)]
    split_points.append((len(document_text), ''))

    for i in range(len(split_points) - 1):
        start, section_name = split_points[i]
        end, _ = split_points[i + 1]
        section_text = document_text[start:end].strip()
        
        if section_name in known_sections:
            subsections = split_into_subsections(section_text, section_name)
            sections.append({
                "section_name": section_name,
                "subsections": subsections
            })
        else:
            # Handle unknown section name (optional)
            print(f"Unknown section name: {section_name}")
    
    return sections

# Function to split section into subsections
def split_into_subsections(section_text, section_name):
    subsections = []
    subsection_patterns = [re.escape(sub) for sub in known_sections[section_name]]
    pattern = re.compile(r'(' + '|'.join(subsection_patterns) + r')', re.IGNORECASE)

    split_points = [(m.start(), m.group()) for m in pattern.finditer(section_text)]
    split_points.append((len(section_text), ''))

    for i in range(len(split_points) - 1):
        start, subsection_name = split_points[i]
        end, _ = split_points[i + 1]
        subsection_text = section_text[start:end].strip()
        
        if subsection_name in known_sections[section_name]:
            subsections.append({
                "subsection_name": subsection_name,
                "subsection_content": subsection_text
            })
        else:
            # Handle unknown subsection name (optional)
            print(f"Unknown subsection name in section {section_name}: {subsection_name}")
    
    return subsections
<div class="file-upload">
  <h2>Upload PDF Document</h2>
  <input type="file" (change)="onFileSelected($event)" accept=".pdf">
  <button (click)="onUpload()" [disabled]="!selectedFile">Upload</button>
</div>
import { Component } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-file-upload',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.css']
})
export class FileUploadComponent {
  selectedFile: File | null = null;
  uploadResponse: string = '';

  constructor(private http: HttpClient) {}

  onFileSelected(event: any): void {
    this.selectedFile = event.target.files[0];
  }

  onUpload(): void {
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile, this.selectedFile.name);

      this.http.post('http://localhost:3000/upload', formData).subscribe(
        response => {
          console.log('File uploaded successfully', response);
          this.uploadResponse = 'File uploaded successfully!';
        },
        (error: HttpErrorResponse) => {
          console.error('File upload failed', error);
          this.uploadResponse = 'File upload failed. Please try again.';
        }
      );
    }
  }
}



