import pdfplumber

def is_heading(element):
    # Check if the element starts with a number and is in bold font
    return element['text'].strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')) and element['fontname'] == 'Arial-BoldMT'

def extract_headings_and_sections(pdf_path):
    headings_and_sections = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract all elements (text and their styling) from the page
            elements = page.extract_words()
            
            # Initialize variables to store section title and associated headings
            section_title = None
            section_headings = []
            
            # Iterate through each element on the page
            for element in elements:
                # Check if the element is a heading
                if is_heading(element):
                    # If section title is found, add previous section's title and headings to the list
                    if section_title and section_headings:
                        headings_and_sections.append({
                            'section_title': section_title,
                            'headings': section_headings
                        })
                    
                    # Set the new section title and clear the list of headings
                    section_title = element['text']
                    section_headings = []
                elif section_title:
                    # If section title is found, consider subsequent elements as headings
                    section_headings.append(element['text'])
            
            # Add the last section's title and headings to the list
            if section_title and section_headings:
                headings_and_sections.append({
                    'section_title': section_title,
                    'headings': section_headings
                })
    
    return headings_and_sections

# Rest of the code for clause extraction remains the same
