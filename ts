from pdfbox import PDFBox
import re

def extract_text_from_pdf(pdf_path):
    pdfbox = PDFBox()
    return pdfbox.extract_text(pdf_path)

def extract_headings_and_clauses(text):
    headings = []
    clauses = []

    # Split the text into lines
    lines = text.split("\n")

    # Extract headings (assuming headings are lines in all caps)
    for line_num, line in enumerate(lines, 1):
        if line.strip().isupper():
            headings.append((line, line_num))  # Add heading and line number

    # Extract clauses associated with each heading
    for i, (heading, line_num) in enumerate(headings):
        if i < len(headings) - 1:
            next_heading_line = headings[i + 1][1]
        else:
            next_heading_line = len(lines) + 1  # Assume end of text if no more headings

        # Extract clauses between current heading and next heading
        clause_lines = lines[line_num:next_heading_line - 1]
        clause = "\n".join(clause_lines).strip()
        clauses.append(clause)

    return headings, clauses

def main():
    pdf_path = "example.pdf"
    text = extract_text_from_pdf(pdf_path)
    headings, _ = extract_headings_and_clauses(text)

    print("Extracted Headings:")
    for idx, (heading, line_num) in enumerate(headings, 1):
        print(f"{idx}. {heading} (Line {line_num})")

    selected_heading_index = int(input("Enter the index of the heading you want to explore: ")) - 1

    if 0 <= selected_heading_index < len(headings):
        _, clauses = extract_headings_and_clauses(text)
        print("Selected Heading Clause:")
        print(clauses[selected_heading_index])
    else:
        print("Invalid heading index. Please enter a valid index.")

if __name__ == "__main__":
    main()
