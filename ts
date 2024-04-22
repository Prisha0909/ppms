def identify_headings(text):
    """
    Identify headings based on the specified condition.
    """
    headings = []
    pattern = r'^\d+\..*?$'  # Assuming headings start with a number followed by a dot
    lines = text.split('\n')
    for line in lines:
        if re.match(pattern, line.strip()):
            headings.append(line.strip())
    return headings

def identify_titles_and_headings(pdf_text):
    """
    Identify titles and headings from the PDF text.
    """
    titles = []
    title_headings_mapping = {}
    current_title = None
    lines = pdf_text.split('\n')
    for line in lines:
        if not current_title:
            current_title = identify_topmost_title(line)
            if current_title:
                titles.append(current_title)
        else:
            headings = identify_headings(line)
            if headings:
                title_headings_mapping[current_title] = headings
                current_title = None
    return titles, title_headings_mapping

def print_titles_and_headings(titles, title_headings_mapping):
    """
    Print titles and their associated headings.
    """
    for title in titles:
        print(f"\nTitle: {title}")
        if title in title_headings_mapping:
            headings = title_headings_mapping[title]
            for heading in headings:
                print(heading)
