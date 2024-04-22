import pdfplumber

def extract_titles_and_headings(pdf_path):
    titles_and_headings = []

    with pdfplumber.open(pdf_path) as pdf:
        current_title = None
        current_headings = []

        for page in pdf.pages:
            # Extract text from the top of the page
            top_text = page.crop((0, 0, page.width, page.height * 0.1)).extract_text()
            
            # Check if the text is in bold (assuming bold text has a fontname containing 'bold')
            bold_text = [text for text in top_text if text['fontname'] and 'bold' in text['fontname'].lower()]
            
            # Extract only the bold text content
            bold_content = [text['text'] for text in bold_text]

            # If bold content is present, it's a new title
            if bold_content:
                # Add the previous title and its headings to the titles_and_headings list
                if current_title:
                    titles_and_headings.append({"title": current_title, "headings": current_headings})
                    current_headings = []
                
                # Set the new title
                current_title = ' '.join(bold_content)
            else:
                # Extract headings using the same logic as before
                headings = identify_headings(page.extract_text())
                if headings:
                    current_headings.extend(headings)

        # Add the last title and its headings to the titles_and_headings list
        if current_title:
            titles_and_headings.append({"title": current_title, "headings": current_headings})

    return titles_and_headings

# Function to identify headings based on the specified condition
def identify_headings(text):
    headings = []
    # Regular expression pattern to match the specified condition (number at the start followed by dot and space)
    pattern = r'^\d+\..*?$'
    lines = text.split('\n')
    for line in lines:
        # Check if the line matches the pattern
        if re.match(pattern, line.strip()):
            headings.append(line.strip())
    return headings

# Test the function
pdf_path = "your_pdf_path.pdf"
titles_and_headings = extract_titles_and_headings(pdf_path)
for item in titles_and_headings:
    print(item["title"])
    for heading in item["headings"]:
        print(heading)
