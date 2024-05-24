import pdfplumber
import re

def extract_text_and_specific_info(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            full_text += text + "\n"

    return full_text

def find_general_info(text):
    # Extract paragraph under the "General" heading
    general_info_pattern = re.compile(r'\nGeneral\s*\n(.*?)(?=\n[A-Z]|$)', re.DOTALL)
    general_info_match = general_info_pattern.search(text)
    
    if general_info_match:
        paragraph = general_info_match.group(1).strip()
        
        # Extract Party A, Party B, and the date using regular expressions
        party_a = re.search(r"'([^']*)'\s*\(\s*\"Party\s*A\"\s*\)", paragraph)
        party_b = re.search(r"'([^']*)'\s*\(\s*\"Party\s*B\"\s*\)", paragraph)
        date = re.search(r'Date\s*:\s*(.*)', paragraph)
        
        party_a = party_a.group(1).strip() if party_a else "Not found"
        party_b = party_b.group(1).strip() if party_b else "Not found"
        date = date.group(1).strip() if date else "Not found"
        
        return {
            "Party A": party_a,
            "Party B": party_b,
            "Date": date
        }
    
    return {
        "Party A": "Not found",
        "Party B": "Not found",
        "Date": "Not found"
    }

def index_paragraphs(text):
    lines = text.split('\n')
    indexed_paragraphs = []
    heading_index = 0
    sub_index = 0
    for line in lines:
        if re.match(r'^\s+[A-Z]', line):  # Identify headings by leading spaces followed by an uppercase letter
            heading_index += 1
            sub_index = 0
            indexed_paragraphs.append(f"{heading_index}. {line.strip()}")
        elif line.strip():  # Non-empty lines considered as paragraphs under a heading
            sub_index += 1
            indexed_paragraphs.append(f"{heading_index}.{sub_index} {line.strip()}")

    return '\n'.join(indexed_paragraphs)

# Example usage
pdf_path = "path/to/your/document.pdf"
full_text = extract_text_and_specific_info(pdf_path)

# Extract and display general information
general_info = find_general_info(full_text)
print("General Information:")
print("Party A:", general_info["Party A"])
print("Party B:", general_info["Party B"])
print("Date:", general_info["Date"])

# Index and display paragraphs
indexed_text = index_paragraphs(full_text)
print("Indexed Paragraphs:")
print(indexed_text)
