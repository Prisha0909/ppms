import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  selectedFile: File;
  documentType: string;
  extractedText: string;
  clauses: any;

  constructor(private http: HttpClient) {}

  onFileChanged(event) {
    this.selectedFile = event.target.files[0];
  }

  onUpload() {
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
  }
}
----------
  <div class="container">
  <div class="row">
    <div class="col">
      <input type="file" (change)="onFileChanged($event)">
      <button (click)="onUpload()">Upload</button>
    </div>
    <div class="col">
      <p>Document Type: {{ documentType }}</p>
    </div>
  </div>

  <div class="row">
    <div class="col">
      <h3>Extracted Text:</h3>
      <div>{{ extractedText }}</div>
    </div>
    <div class="col">
      <h3>Predicted Clauses:</h3>
      <div *ngIf="clauses">
        <p>Section: {{ clauses.section }}</p>
        <p>Clause: {{ clauses.clause }}</p>
        <p *ngIf="clauses.sub_section">Sub-section: {{ clauses.sub_section }}</p>
      </div>
    </div>
  </div>
</div>
