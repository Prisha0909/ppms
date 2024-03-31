import { Component } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

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

    // Set the headers for the HTTP request
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    // Send the HTTP POST request
    this.http.post(apiUrl, data, { headers })
      .pipe(
        catchError(this.handleError)
      )
      .subscribe(
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

  // Error handling function
  private handleError(error: HttpErrorResponse) {
    let errorMessage = 'Unknown error occurred';
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }
    console.error(errorMessage);
    return throwError(errorMessage);
  }
}
