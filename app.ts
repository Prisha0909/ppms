<div class="extracted-text">
  <div *ngFor="let paragraph of paragraphs" class="paragraph">
    <p>{{ paragraph.text }}</p>
    <p><strong>Predicted Clause:</strong> {{ paragraph.clause }}</p>
  </div>
</div>
------------
    import { Component, OnInit } from '@angular/core';
import { DocumentService } from '../document.service';

@Component({
  selector: 'app-extracted-text',
  templateUrl: './extracted-text.component.html',
  styleUrls: ['./extracted-text.component.css']
})
export class ExtractedTextComponent implements OnInit {
  paragraphs: any[] = [];

  constructor(private documentService: DocumentService) { }

  ngOnInit(): void {
    this.documentService.getDocumentData().subscribe(data => {
      if (data) {
        this.paragraphs = data.clauses;
      }
    });
  }
}

------------
  .extracted-text {
  clear: both;
  margin-top: 20px;
  padding-left: 20px;
}

.paragraph {
  margin-bottom: 20px;
}
-----------
  
