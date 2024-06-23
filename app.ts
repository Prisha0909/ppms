<div class="pdf-uploader">
  <label class="custom-file-upload">
    <input type="file" (change)="onFileSelected($event)" />
    Upload PDF
  </label>
</div>
-------
    import { Component } from '@angular/core';
import { DocumentService } from '../document.service';

@Component({
  selector: 'app-pdf-uploader',
  templateUrl: './pdf-uploader.component.html',
  styleUrls: ['./pdf-uploader.component.css']
})
export class PdfUploaderComponent {

  constructor(private documentService: DocumentService) { }

  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    if (file) {
      this.documentService.uploadPdf(file).subscribe(response => {
        console.log(response);  // For debugging
        this.documentService.setDocumentData(response);
      });
    }
  }
}
-----------
    .pdf-uploader {
  margin-top: 20px;
}

.custom-file-upload {
  display: inline-block;
  padding: 10px 20px;
  cursor: pointer;
  background-color: #007bff;
  color: white;
  border-radius: 5px;
  border: 1px solid #007bff;
  text-align: center;
}

.custom-file-upload:hover {
  background-color: #0056b3;
}
------------
    <div class="doc-type-display">
  <h3>Document Type: {{ docType }}</h3>
</div>

    ---------
    import { Component, OnInit } from '@angular/core';
import { DocumentService } from '../document.service';

@Component({
  selector: 'app-doc-type-display',
  templateUrl: './doc-type-display.component.html',
  styleUrls: ['./doc-type-display.component.css']
})
export class DocTypeDisplayComponent implements OnInit {
  docType: string;

  constructor(private documentService: DocumentService) { }

  ngOnInit(): void {
    this.documentService.getDocumentData().subscribe(data => {
      if (data) {
        this.docType = data.doc_type;
      }
    });
  }
}
---
    .doc-type-display {
  float: right;
  margin-top: 20px;
  margin-right: 20px;
}
--------
    
