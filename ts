import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def identify_headings_and_clauses(text):
    # Implement logic to identify headings and associated clauses
    # For example, you can use regular expressions to find patterns indicating headings
    headings = [line.strip() for line in text.split("\n") if line.strip().isupper()]
    clauses = []
    for heading in headings:
        start_index = text.find(heading)
        next_heading_index = text.find("\n", start_index + 1)
        if next_heading_index == -1:
            clause = text[start_index:]
        else:
            clause = text[start_index:next_heading_index]
        clauses.append(clause)
    return headings, clauses

def main():
    pdf_path = "Shareholders-Agreement-Template.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    headings, clauses = identify_headings_and_clauses(extracted_text)
    
    print("Extracted Headings:")
    for idx, heading in enumerate(headings):
        print(f"{idx + 1}. {heading}")

    selected_heading_index = int(input("Enter the index of the heading you want to explore: ")) - 1
    selected_heading = headings[selected_heading_index]

    highlighted_clause = clauses[selected_heading_index]
    # Implement logic to highlight or underline the clause
    # For example, you can add HTML tags to highlight or underline the text

    print("Highlighted Clause:")
    print(highlighted_clause)

if __name__ == "__main__":
    main()
