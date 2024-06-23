import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  selectedFile: File | null = null;
  documentType: string = '';
  extractedText: string = '';
  clauses: { section: string, clause: string, sub_section: string | null } | null = null;

  constructor(private http: HttpClient) {}

  onFileChanged(event: any) {
    this.selectedFile = event.target.files[0];
  }

  onUpload() {
    if (this.selectedFile) {
      const uploadData = new FormData();
      uploadData.append('file', this.selectedFile, this.selectedFile.name);

      this.http.post<any>('http://localhost:5000/upload-pdf', uploadData)
        .subscribe(
          (response) => {
            this.documentType = response.doc_type;
            this.extractedText = response.text;
            this.clauses = response.clauses;
          },
          (error) => {
            console.error('Error uploading file', error);
          }
        );
    } else {
      console.error('No file selected');
    }
  }
}
