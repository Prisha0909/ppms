def identify_headings(text):
    headings = []
    # Regular expression pattern to match the specified condition (number at the start followed by bold font)
    pattern = r'^\d+\..*?$'  # This assumes headings start with a number followed by a dot
    lines = text.split('\n')  # Split the text into lines
    for line in lines:
        # Check if the line matches the pattern
        if re.match(pattern, line.strip()):
            headings.append(line.strip())
    return headings
def present_headings(headings):
    for idx, heading in enumerate(headings):
        print(f"{idx + 1}. {heading}")

def extract_clause(selected_heading, headings, text):
    selected_heading_text = headings[selected_heading - 1]
    start_index = text.find(selected_heading_text)
    next_heading_index = len(text)
    for heading in headings[selected_heading:]:
        heading_index = text.find(heading)
        if heading_index != -1 and heading_index < next_heading_index:
            next_heading_index = heading_index
    clause_text = text[start_index:next_heading_index]
    return clause_text

def main():
    pdf_path = "path/to/your/pdf/document.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    headings = identify_headings(extracted_text)
    present_headings(headings)
    selected_heading = int(input("Enter the index of the heading you want to explore: "))
    clause = extract_clause(selected_heading, headings, extracted_text)
    print("Clause associated with the selected heading:")
    print(clause)

if __name__ == "__main__":
    main()
