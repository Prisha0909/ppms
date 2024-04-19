
import pdfplumber
import spacy
from collections import defaultdict
import string
import re
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text using spaCy
def preprocess_text(text):
    return text

# Function to extract text from PDF and preprocess it
def extract_and_preprocess_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text()
    preprocessed_text = preprocess_text(all_text)
    return preprocessed_text

# Function to extract headings with page number, bounding box, and block number


# Function to extract headings with page number, bounding box, and block number
def extract_headings_with_info(pdf_path):
    headings_info = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            for block_num, block in enumerate(page.extract_words(), 1):
                if re.match(r'^\d+\.', block["text"].strip()):  # Check if the text starts with a digit followed by a period
                    bbox = (block["x0"], block["top"], block["x1"], block["bottom"])  # Extract bounding box
                    heading_info = {
                        "heading": block["text"].strip(),
                        "page_num": page_num,
                        "bounding_box": bbox,
                        "block_num": block_num
                    }
                    headings_info.append(heading_info)
    return headings_info


# Function to get the text associated with a selected heading
def get_associated_text(selected_heading_index, headings_info, pdf_path):
    selected_heading = headings_info[selected_heading_index]
    selected_page_num = selected_heading['page_num']
    selected_bbox = selected_heading['bounding_box']
    associated_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            if page_num < selected_page_num:
                continue
            elif page_num == selected_page_num:
                for block_info in headings_info[selected_heading_index:]:
                    if block_info['page_num'] == page_num:
                        break
                    associated_text += block_info['heading'] + "\n"
                for block in page.extract_words():
                    block_bbox = (block["x0"], block["top"], block["x1"], block["bottom"])
                    if selected_bbox[1] < block_bbox[3] < selected_bbox[3]:
                        associated_text += block["text"]
            else:
                break  # Stop processing pages once we move past the selected heading's page
    return associated_text

# Main function
def main():
    pdf_path = "new testing.pdf"

    # Extracting text from PDF
    text = extract_and_preprocess_text(pdf_path)
    print("Extracted Text:")
    print(text)

    # Extracting headings with their information
    headings_info = extract_headings_with_info(pdf_path)
    print("\nExtracted Headings:")
    for idx, heading_info in enumerate(headings_info):
        print(f"{idx + 1}. {heading_info['heading']} - Page: {heading_info['page_num']}, Bounding Box: {heading_info['bounding_box']}, Block: {heading_info['block_num']}")

    # Selecting a heading
    selected_heading_index = int(input("Enter the index of the heading you want to explore: ")) - 1
    if 0 <= selected_heading_index < len(headings_info):
        selected_heading = headings_info[selected_heading_index]
        print("\nSelected Heading:", selected_heading["heading"])
        # Extracting text associated with the selected heading
        associated_text = get_associated_text(selected_heading_index, headings_info, pdf_path)
        print("Text Associated with the Selected Heading:")
        print(associated_text.strip())
    else:
        print("Invalid heading index. Please enter a valid index.")

if __name__ == "__main__":
    main()
