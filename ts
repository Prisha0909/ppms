import jpype
from jpype import JClass, JString, getDefaultJVMPath

def extract_text_from_pdf(pdf_path):
    jpype.startJVM(getDefaultJVMPath())
    PDFTextStripper = JClass("org.apache.pdfbox.text.PDFTextStripper")
    PDDocument = JClass("org.apache.pdfbox.pdmodel.PDDocument")
    FileInputStream = JClass("java.io.FileInputStream")

    doc = PDDocument(FileInputStream(pdf_path))
    stripper = PDFTextStripper()

    text = []
    for page in range(doc.getNumberOfPages()):
        stripper.setStartPage(page + 1)
        stripper.setEndPage(page + 1)
        text.append(stripper.getText(doc))

    doc.close()
    jpype.shutdownJVM()

    return text

def extract_headings_from_text(text):
    headings = []
    for page_num, page_text in enumerate(text, 1):
        lines = page_text.split("\n")
        for line_num, line in enumerate(lines, 1):
            if line.strip().isupper():
                headings.append((line.strip(), page_num, line_num))

    return headings

def extract_clause_for_heading(text, heading):
    clauses = []
    page_num, _, heading_line_num = heading

    lines = text[page_num - 1].split("\n")
    for line_num in range(heading_line_num + 1, len(lines)):
        line = lines[line_num - 1]
        if line.strip().isupper():
            break
        clauses.append(line)

    return "\n".join(clauses)

def main():
    pdf_path = "example.pdf"
    text = extract_text_from_pdf(pdf_path)
    headings = extract_headings_from_text(text)

    print("Extracted Headings:")
    for idx, (heading, page_num, line_num) in enumerate(headings, 1):
        print(f"{idx}. {heading} - Page: {page_num}, Line: {line_num}")

    selected_heading_idx = int(input("Enter the index of the heading you want to explore: ")) - 1
    if 0 <= selected_heading_idx < len(headings):
        selected_heading = headings[selected_heading_idx]
        clause = extract_clause_for_heading(text, selected_heading)
        print(f"\nClause for {selected_heading[0]}:")
        print(clause)
    else:
        print("Invalid heading index. Please enter a valid index.")

if __name__ == "__main__":
    main()
