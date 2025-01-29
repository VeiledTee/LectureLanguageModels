import os
from docling.document_converter import DocumentConverter
import re

def pdf_to_markdown(pdf_path, output_txt_path):
    # Initialize the converter
    doc_converter = DocumentConverter()

    # Convert PDF to markdown
    result = doc_converter.convert(pdf_path)
    markdown_text = result.document.export_to_markdown()

    processed_lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]

    # Save as a text file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            line.replace("<!-- image -->", '')
            if line != '' or line != '\n':
                f.write(line + '\n')

    print(f"Converted {pdf_path} to Markdown and saved as {output_txt_path}")


def extract_questions(extract_from):
    extracted_questions = []
    for line in extract_from.split('\x1E'):  # split by delimiter
        for query in line.split('\x1F'):  # split questions
            if line != '':
                extracted_questions.append(query.strip())
    return extracted_questions


def load_markdown_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    # Splitting sections by headers (##) while keeping them
    split_sections = re.split(r'(## .*\n)', text)
    extracted_sections = {}
    for i in range(1, len(split_sections), 2):
        section_header = split_sections[i].strip("# \n")
        section_content = split_sections[i + 1].strip()
        # Replace only actual section breaks with \x1E, keep the rest as is
        extracted_sections[section_header] = section_content.replace('\n', ' ') + '\x1E'
    return extracted_sections


# Directory containing the PDF files
# dir_path = "/home/penguins/Documents/PhD/Education RA/AI Course/Exams"
dir_path = "/home/penguins/Documents/PhD/Education RA/AI Course/Lecture Notes"

# Create a converter instance
converter = DocumentConverter()

for filename in os.listdir(dir_path):
    source_path = os.path.join(dir_path, filename)

    if filename.endswith('.pdf'):
        output_path = os.path.join(dir_path, filename.split('.')[0] + '_parsed.txt')
        pdf_to_markdown(source_path, output_path)

    elif filename.endswith('_answerless.txt'):
        with (open(source_path, 'r', encoding='utf-8') as f):
            input_text = f.read()
            sections = load_markdown_sections('AI Course/Exams/q1_soln_answerless.txt')

            # Print all sections
            for header, content in sections.items():
                if len(content) >= len(header):
                    questions = extract_questions(content)
                    print(f"## {header}: {questions}")
