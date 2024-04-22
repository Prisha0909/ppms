import pdfplumber

def extract_headings_and_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        titles_and_headings = {}
        current_title = None
        
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                # Check if the line is a title (topmost and in bold)
                if i == 0 and (line.isupper() or (line[0].isupper() and line[1:].islower())):
                    current_title = line.strip()
                    # If this is a new title, start a new list of headings
                    if current_title not in titles_and_headings:
                        titles_and_headings[current_title] = []
                # Check if the line is a heading (starts with a number followed by a dot)
                elif current_title and line.strip().startswith(tuple(str(i) for i in range(10))) and '.' in line:
                    heading = line.strip().split(' ', 1)[1]
                    titles_and_headings[current_title].append((heading, []))
                elif current_title:
                    # Add text to the last heading under the current title
                    if titles_and_headings[current_title]:
                        titles_and_headings[current_title][-1][1].append(line.strip())
        
        return titles_and_headings

# Example usage
pdf_path = 'path_to_your_pdf.pdf'
titles_and_headings = extract_headings_and_text(pdf_path)

# You can then display the titles and headings as before, and get user selection.
