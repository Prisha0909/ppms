import pdfplumber

def is_section_title(element):
    # Check if the element is at the top of the page, in the center, and in bold font
    return (element['top'] < 50) and (element['x0'] < 100) and (element['x1'] > 400) and (element['fontname'] == 'Arial-BoldMT')

def is_heading(element):
    # Check if the element starts with a number and is in bold font
    return element['text'].strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')) and element['fontname'] == 'Arial-BoldMT'

def extract_headings_and_sections(pdf_path):
    headings_and_sections = []
    
    with pdfplumber.open(pdf_path) as pdf:
        section_title = None
        section_headings = []
        
        for page in pdf.pages:
            # Extract all elements (text and their styling) from the page
            elements = page.extract_words()
            
            # Iterate through each element on the page
            for element in elements:
                # Check if the element is a section title
                if is_section_title(element):
                    # If a new section title is found, add the previous section and its headings to the list
                    if section_title:
                        headings_and_sections.append({
                            'section_title': section_title,
                            'headings': section_headings
                        })
                    
                    # Set the new section title and clear the list of headings
                    section_title = element['text']
                    section_headings = []
                elif is_heading(element):
                    # If element is a heading, add it to the list of headings
                    section_headings.append(element['text'])
            
            # Add the headings from the current page to the current section
            if section_headings:
                headings_and_sections.append({
                    'section_title': section_title,
                    'headings': section_headings
                })
    
    return headings_and_sections

def print_headings_and_sections(headings_and_sections):
    for section in headings_and_sections:
        print(f"Section Title: {section['section_title']}")
        print("Headings:")
        for heading in section['headings']:
            print(f"- {heading}")
        print()

def main():
    pdf_file_path = "your_pdf_file.pdf"  # Replace with the path to your PDF file
    headings_and_sections = extract_headings_and_sections(pdf_file_path)
    print_headings_and_sections(headings_and_sections)

if __name__ == "__main__":
    main()
