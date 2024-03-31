import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  constructor(private http: HttpClient) {}

  // Function to send data to Streamlit backend
  sendDataToStreamlit(data: any): void {
    const apiUrl = 'http://localhost:8501/process_data'; // URL of the Streamlit endpoint
    this.http.post(apiUrl, data).subscribe(
      response => {
        console.log('Response from Streamlit:', response);
      },
      error => {
        console.error('Error sending data to Streamlit:', error);
      }
    );
  }

  // Function to handle file upload and send data to Streamlit
  onFileSelected(event: any): void {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = () => {
      const fileContent = reader.result as string;
      this.sendDataToStreamlit({ fileContent });
    };
    reader.readAsText(file);
  }
}
