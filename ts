import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent {
  selectedFile: File;
  extractedText: string;
  documentType: string;

  constructor(private http: HttpClient) { }

  onFileSelected(event: any): void {
    this.selectedFile = event.target.files[0];
  }

  uploadFile(): void {
    if (!this.selectedFile) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.http.post<any>('http://localhost:3000/upload', formData).subscribe(
      (res) => {
        console.log('File uploaded successfully:', res);
        // Fetch extracted text
        this.fetchExtractedText();
        // Fetch document type
        this.fetchDocumentType();
      },
      (err) => {
        console.error('Error uploading file:', err);
        // Handle errors
      }
    );
  }

  fetchExtractedText(): void {
    this.http.get<any>('http://localhost:3000/extracted-text').subscribe(
      (res) => {
        console.log('Extracted text:', res);
        this.extractedText = res.extractedText;
      },
      (err) => {
        console.error('Error fetching extracted text:', err);
        // Handle errors
      }
    );
  }

  fetchDocumentType(): void {
    this.http.get<any>('http://localhost:3000/document-type').subscribe(
      (res) => {
        console.log('Document type:', res);
        this.documentType = res.documentType;
      },
      (err) => {
        console.error('Error fetching document type:', err);
        // Handle errors
      }
    );
  }
}
