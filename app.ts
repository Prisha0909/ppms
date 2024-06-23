import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DocumentService {
  private documentData = new BehaviorSubject<any>(null);

  constructor(private http: HttpClient) { }

  uploadPdf(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<any>('http://localhost:5000/upload-pdf', formData);
  }

  setDocumentData(data: any): void {
    this.documentData.next(data);
  }

  getDocumentData(): Observable<any> {
    return this.documentData.asObservable();
  }
}
----
  <div class="app">
  <div class="top-section">
    <app-pdf-uploader></app-pdf-uploader>
    <app-doc-type-display></app-doc-type-display>
  </div>
  <app-extracted-text></app-extracted-text>
</div>
-
  import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
}
-----
  Module Class (app.module.ts):
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';

import { AppComponent } from './app.component';
import { PdfUploaderComponent } from './pdf-uploader/pdf-uploader.component';
import { DocTypeDisplayComponent } from './doc-type-display/doc-type-display.component';
import { ExtractedTextComponent } from './extracted-text/extracted-text.component';
import { DocumentService } from './document.service';

@NgModule({
  declarations: [
    AppComponent,
    PdfUploaderComponent,
    DocTypeDisplayComponent,
    ExtractedTextComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule
  ],
  providers: [DocumentService],
  bootstrap: [AppComponent]
})
export class AppModule { }
