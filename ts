import pdfplumber

def extract_headings(pdf_path):
    headings = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split("\n")
            for line in lines:
                if line.strip() and line.strip()[0].isdigit() and line.strip().endswith("."):
                    headings.append(line.strip())
    return headings

def extract_clause(pdf_path, selected_heading):
    clause = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split("\n")
            found_heading = False
            for line in lines:
                if line.strip() == selected_heading:
                    found_heading = True
                    continue  # Skip the heading line itself
                if found_heading:
                    if line.strip():  # Skip empty lines
                        clause += line.strip() + "\n"  # Add the line to the clause
                if found_heading and line.strip() and line.strip()[0].isdigit() and line.strip().endswith("."):
                    break  # Stop extracting text if a new heading is found
    return clause.strip()

def main():
    pdf_path = "path/to/your/pdf_file.pdf"
    headings = extract_headings(pdf_path)

    print("Available Headings:")
    for idx, heading in enumerate(headings, 1):
        print(f"{idx}. {heading}")

    choice = int(input("Enter the index of the heading you want to explore: "))
    if 1 <= choice <= len(headings):
        selected_heading = headings[choice - 1]
        clause = extract_clause(pdf_path, selected_heading)
        print("\nClause under selected heading:")
        print(clause)
    else:
        print("Invalid choice. Please enter a valid index.")

if __name__ == "__main__":
    main()
