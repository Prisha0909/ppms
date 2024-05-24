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
    # Use regular expressions to identify the heading "General" and extract the first paragraph under it
    general_info = re.search(r'(?<=\nGeneral\s*\n)(.*?)(?=\n[A-Z])', text, re.DOTALL)
    
    if general_info:
        paragraph = general_info.group(1).strip()
        
        # Use regular expressions to extract Party A, Party B, and the date
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

# Example usage
pdf_path = "path/to/your/document.pdf"
full_text = extract_text_and_specific_info(pdf_path)
general_info = find_general_info(full_text)
print("General Information:")
print("Party A:", general_info["Party A"])
print("Party B:", general_info["Party B"])
print("Date:", general_info["Date"])
